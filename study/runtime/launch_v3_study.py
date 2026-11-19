"""Separate fixed1500 recovery cohort on twoT4s; no model/Torch/work on import.

Development: fixed B teacher, CPU autograd prerequisites, active pilots, BEFORE-
validation common-budget decision, then full fixed grid. Confirmation requires
an independently frozen validation-only decision. No score parsing or retries.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path('/kaggle/working/generative_recovery_2500')
STUDY='generative_recovery_1500'
SOURCES=['models.py','model_losses.py','compensated_interaction.py','train_v3.py',
         'interaction_witness.py','interaction_statistics.py']
AUX=['src/prepare_teacher_v3.py','runtime/archive_v3_model.py','tests/test_interaction_statistics.py',
     'tests/test_interaction_witness_torch.py','runtime/launch_v3_study.py']
CACHE_SHA='98d39e0febb8c1946aba70eca2ca5cb96a657cdf6075f7ae32fc56d4c6f9bd5c'
_STOP=threading.Event()
GRID=[('C_l0p1_k0','C',.1,0.),('C_l0p1_k1','C',.1,1.),
      ('C_l1_k0','C',1.,0.),('C_l1_k1','C',1.,1.)]

def require(value,message):
    if not value:raise RuntimeError(message)

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):digest.update(block)
    return digest.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def source_map(root):return {name:sha(root/'src'/name) for name in SOURCES}
def aux_map(root):return {name:sha(root/name) for name in AUX}

def data_check(root):
    import numpy as np
    ready_path=root/'data/data_ready.json';r=read(ready_path)
    require(r['status']=='frozen_clean_component_split' and r['study_id'] in ('generative_recovery_2500','generative_recovery_1500')
        and r['prior_definition_ready_sha256']=='62347bb0d64d5d8fb816b8c233cdda41be30f89bb92d0a86bc30264f5f3ee639'
        and r['reconstructed_inputs_only'] is True and r['test_quality_accessed'] is False,
        'Complete approved clean component split required')
    index=root/'data/clean_index.npz';components=root/'data/component_ids.npy'
    cache=Path(r['cache_path'])
    require(not any(p.is_symlink() for p in [ready_path,index,components,cache]),'Input aliases refused')
    require(sha(index)==r['clean_index_sha256'] and sha(components)==r['component_ids_sha256']
        and sha(cache)==r['original_cache_sha256']==CACHE_SHA,'Actual data bytes differ')
    with np.load(index,allow_pickle=False) as x:
        require(set(x.files)=={'filenames','groups','splits'},'Wrong clean index arrays')
        names,groups,splits=x['filenames'],x['groups'],x['splits']
    c=np.load(components,allow_pickle=False)
    require(names.shape==groups.shape==splits.shape==c.shape==(202599,)
        and c.dtype==np.int64 and np.isin(splits,[-1,0,1,2]).all()
        and np.isin(groups,[0,1,2,3]).all(),'Wrong clean input layout')
    sets=[set(c[splits==s].tolist()) for s in range(3)]
    require(all(not sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)]),'Components cross new splits')
    counts=[[int(np.sum((splits==s)&(groups==g))) for g in range(4)] for s in range(3)]
    cluster_counts=[[len(set(c[(splits==s)&(groups==g)].tolist())) for g in range(4)] for s in range(3)]
    require(all(v>=100 for row in counts[1:] for v in row)
        and all(v>=2 for row in cluster_counts for v in row),'Frozen support floor failed')
    require(r['split_group_image_counts']==counts and r['split_group_component_counts']==cluster_counts,
        'Ready aggregate counts differ')
    require(r['original_val_test_images_used'] is False,'Fresh original-train-only cohort required')
    return r,sha(ready_path),counts

def common(root,ready,ready_sha,counts,sources):
    return {'study_id':STUDY,'cache_path':ready['cache_path'],'cache_sha256':CACHE_SHA,
        'metadata_npz':str(root/'data/clean_index.npz'),'index_sha256':ready['clean_index_sha256'],
        'component_ids_path':str(root/'data/component_ids.npy'),
        'component_ids_sha256':ready['component_ids_sha256'],'data_ready_sha256':ready_sha,
        'batch_size':64,'latent_dim':128,'base_channels':32,'learning_rate':.0002,
        'adam_betas':[0.,.9],'ema_decays':[.99,.999],'amp':True,'amp_initial_scale':1024,
        'gsr_beta':.999,'train_group_counts':counts[0],'shared_latent_g_batch':True,
        'teacher_feature_dim':128,'fake_ema_beta':.9,'witness_warmup_successful_updates':500,
        'scale_floor':.001,'pretrained_generator_or_discriminator':False,
        'quality_metrics_during_training':False,'source_sha256':sources,
        'log_every':500,'checkpoint_every':1500,'timing_warmup':100,'cpu_threads':2}

def clock(root):
    return datetime.fromisoformat(read(root/'study_clock.json')['deadline_utc'])
def remaining(root):return (clock(root)-datetime.now(timezone.utc)).total_seconds()

def stop_owned(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=5)

def run_child(root,command,device,log_path,cap):
    require(math.isfinite(cap) and cap>0,'No positive child budget remains')
    require(not _STOP.is_set(),'Study stop already requested')
    started=time.monotonic();process=None;timed_out=False
    try:
        with Path(log_path).open('x') as log:
            process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(device),
                    'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            while process.poll() is None:
                if _STOP.is_set():
                    stop_owned(process)
                    raise RuntimeError('Shared study stop requested; owned child stopped')
                if time.monotonic()-started>=cap:
                    timed_out=True;stop_owned(process);break
                time.sleep(min(.5,max(.01,cap-(time.monotonic()-started))))
            code=process.wait(timeout=5)
    except BaseException:
        if process is not None:stop_owned(process)
        raise
    result={'return_code':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-started,
        'cap_seconds':cap,'log_sha256':sha(log_path),'gpu_visible_device':device}
    write(str(log_path)+'.process.json',result)
    require(code==0 and not timed_out,'Owned child failed; no automatic retry')
    return result

def train(root,config,device,reserve=0):
    config_path=root/'configs'/config['phase']/(config['run_id']+'.json');write(config_path,config)
    log=root/'logs'/(config['phase']+'_'+config['run_id']+'.log');log.parent.mkdir(exist_ok=True)
    run_child(root,[sys.executable,'-B',str(root/'src/train_v3.py'),'--config',str(config_path),
        '--root',str(root)],device,log,max(0,remaining(root)-reserve))
    folder=root/config['phase']/config['run_id'];record=read(folder/'complete.json')
    require(record['status']=='complete' and record['config']==config and
        record['completed_steps']==config['steps'] and record['arm']==config['arm'] and
        record['seed']==config['seed'] and record['real_images']==config['steps']*64
        and record['group_exposures']==[config['steps']*16]*4,'Actual completion differs')
    require(record['generator_parameters']==737315 and record['discriminator_parameters']==888545,
        'Actual primary backbone changed')
    if config['phase']!='pilot':
        require(sha(folder/'latest.pt')==record['checkpoint_sha256'] and
            sha(folder/'D_final.pt')==record['D_final_sha256'],'Actual checkpoint bytes changed')
        import numpy as np
        e=np.load(folder/'exposures.npy',allow_pickle=False)
        with np.load(config['metadata_npz'],allow_pickle=False) as data:splits=data['splits']
        require(e.shape==(202599,) and e.dtype==np.int64 and bool((e>=0).all())
            and int(e.sum())==config['steps']*64 and not bool((e[splits!=0]>0).any()),
            'Actual heldout/exposure assertion failed')
        require(set(record['ema_artifacts'])=={'0.99','0.999'},'Both fixed G EMAs required')
        for decay,item in record['ema_artifacts'].items():
            require(Path(item['path'])==folder/('EMA_'+decay+'.pt')
                and sha(item['path'])==item['sha256'],'EMA output changed')
    if config['arm']=='C':
        require(record['witness_updates']==record['successful_g_updates'],'Tracker did not follow successfulG')
        if config['phase']=='pilot':require(record['witness_updates']>=550,'Pilot did not exercise active witness')
    if config['phase']=='confirmation':
        # Recover each selected inference EMA before this lane starts its next
        # model; the independent GPU lane can continue training concurrently.
        run_child(root,[sys.executable,'-B',str(root/'runtime/archive_v3_model.py'),
            '--root',str(root),'--run',config['run_id']],device,
            root/'logs'/('archive_'+config['run_id']+'.log'),min(60,remaining(root)))
        durable_path=root/'durable'/config['run_id']/'complete.json';durable=read(durable_path)
        require(durable['status']=='durable_selected_EMA_archive_and_text_chunks_verified'
            and durable['run']==config['run_id']
            and durable['selected_ema_decay']==config['selected_ema_decay']
            and durable['EMA_sha256']==record['ema_artifacts'][str(config['selected_ema_decay'])]['sha256']
            and durable['source_sha256']==sha(root/'runtime/archive_v3_model.py'),
            'Immediate durable selected-EMA archive binding differs')
        record['durable_complete_sha256']=sha(durable_path)
    return record

def with_run(base,name,arm,coefficient,kappa,phase,seed,steps):
    return {**base,'run_id':name+'_seed'+str(seed),'arm':arm,'coefficient':coefficient,
        'kappa':kappa,'phase':phase,'seed':seed,'steps':steps}

def teacher_binding(root,base):
    done=read(root/'teacher/complete.json')
    require(done['status']=='complete' and done['source_plan_sha256']==sha(root/'study_plan_frozen.json'),
        'Teacher not bound to prospective plan')
    return {**base,'teacher_artifact_path':str(root/'teacher/fixed_teacher.pt'),
        'teacher_artifact_sha256':done['teacher_artifact_sha256'],
        'teacher_statistics_path':str(root/'teacher/statistics.json'),
        'teacher_statistics_sha256':done['statistics_sha256']}

def balanced_schedule(jobs,speeds):
    lanes=[[],[]];loads=[0.,0.]
    ranked=sorted(enumerate(jobs),key=lambda item:(-item[1]['steps']*speeds[item[1]['arm']],item[0]))
    for _,job in ranked:
        lane=0 if loads[0]<=loads[1] else 1
        lanes[lane].append(job);loads[lane]+=job['steps']*speeds[job['arm']]
    return lanes,max(loads)

def run_parallel(root,jobs,speeds,reserve):
    lanes,_=balanced_schedule(jobs,speeds);results=[];errors=[]
    def lane_work(device,lane):
        out=[]
        for config in lane:out.append(train(root,config,device,reserve=reserve))
        return out
    # A main-thread interrupt sets this event BEFORE executor.__exit__ waits.
    # Every worker polls it within .5s and stops only its own created group.
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[pool.submit(lane_work,i,lane) for i,lane in enumerate(lanes)]
        try:
            for item in as_completed(pending):
                try:results.extend(item.result())
                except Exception as error:
                    errors.append(type(error).__name__+': '+str(error));_STOP.set()
        except BaseException:
            _STOP.set()
            raise
    require(not errors,'One or more fixed lanes failed: '+'; '.join(errors))
    return results

def development(root):
    require(not any((root/name).exists() for name in ['study_clock.json','study_plan_frozen.json',
        'development','pilot','teacher','development_summary.json']),'Preserve prior phase; no automatic restart')
    ready,ready_sha,counts=data_check(root);sources=source_map(root);aux=aux_map(root)
    approval_path=root/'training_sources_approved.json';approval=read(approval_path)
    require(approval['status']=='approved_before_v3_teacher_grid_validation_and_test'
        and approval['study_id']==STUDY and approval['training_sources']==sources
        and approval['auxiliary_sources']==aux and approval['data_ready_sha256']==ready_sha
        and approval['test_quality_seen'] is False and approval['ai_root_approval'] is True
        and approval['max_study_seconds']==6000,'Exact prospective source/data/budget approval required')
    approved_deadline=datetime.fromisoformat(approval['deadline_utc'].replace('Z','+00:00'))
    require(approved_deadline.tzinfo is not None,'Timezone-aware absolute approved deadline required')
    require(approved_deadline==datetime(2026,11,19,11,53,tzinfo=timezone.utc),'Absolute approved recovery deadline required')
    deadline=approved_deadline
    write(root/'study_clock.json',{'started_utc':utc(),'deadline_utc':deadline.isoformat(),
        'approval_sha256':sha(approval_path),'includes_teacher_grid_confirmation_evaluation_and_export':True})
    plan={'status':'frozen_before_teacher_grid_validation_test','study_id':STUDY,
        'training_sources':sources,'auxiliary_sources':aux,'data_ready_sha256':ready_sha,
        'data_approval_sha256':sha(approval_path),'quality_results_seen':False,
        'development_seed':909,'development_steps':1500,'grid':GRID,
        'ema_decays':[.99,.999],'main_seeds':[101,202,303],
        'main_steps_rule':'FIXED1500_BEFORE_ANY_VALIDATION_no_budget_selection',
        'teacher':'last_rawD_B_seed909_1500_fixed_eval_fulltrain_stats',
        'main_optional_V2':False,'maximum_confirmation_runs':9,
        'fixed_main_steps_before_any_validation':1500,'V2_lost_models_or_metrics_pooled':False,
        'selection':'validation_group11_KID_min_rounded12_then_smallerlambda_then_kappa0_then_ema.999',
        'evaluation':{'generated_per_group':512,'analysis_per_group':512,'real_cap_per_group':512,
            'generated_latent_coupling':'shared_across_four_conditions_and_models',
            'latent_bank_algorithm':'numpy_default_rng_PCG64_standard_normal_float32_512x128',
            'validation_z_seed':64001,'validation_real_hash_seed':65001,
            'test_z_seed':94001,'test_real_hash_seed':95001,'bootstrap_seed':95003,
            'bootstrap_replicates':2000,'family_size':2,'conservative_reserved_slots':4,'one_sided_percentile_level':.9875,
            'real_bootstrap_unit':'leakage_union_component','training_seed_population_inference':False}}
    write(root/'study_plan_frozen.json',plan)
    base=common(root,ready,ready_sha,counts,sources)
    (root/'logs').mkdir(exist_ok=True)
    probe=('import json,platform,torch,numpy as np;'
        'assert torch.cuda.is_available() and torch.cuda.device_count()==2;'
        'names=[torch.cuda.get_device_name(i) for i in range(2)];'
        'assert all("T4" in name for name in names);'
        'print(json.dumps({"gpus":names,"torch":torch.__version__,"numpy":np.__version__,'
        '"python":platform.python_version(),"cuda":torch.version.cuda}))')
    run_child(root,[sys.executable,'-B','-c',probe],'0,1',root/'logs/GPU_environment.log',min(60,remaining(root)))
    run_child(root,[sys.executable,'-B','-m','unittest','discover','-s','tests',
        '-p','test_interaction*.py','-v'],0,root/'logs/CPU_autograd_prerequisites.log',min(120,remaining(root)))
    b=train(root,with_run(base,'B','B',0.,0.,'development',909,1500),0,reserve=1200)
    run_child(root,[sys.executable,'-B',str(root/'src/prepare_teacher_v3.py'),'--root',str(root)],
        0,root/'logs/prepare_teacher.log',min(600,remaining(root)-1800))
    base=teacher_binding(root,base)
    pilot=with_run({**base,'timing_warmup':550},'C_active','C',1.,1.,'pilot',707,600)
    c=train(root,pilot,0,reserve=1500)
    speeds={'B':b['measured_seconds_per_step'],'C':c['measured_seconds_per_step']}
    require(all(math.isfinite(v) and v>0 for v in speeds.values()),'Bad throughput')
    jobs=[with_run(base,n,a,l,k,'development',909,1500) for n,a,l,k in GRID]
    _,dev_seconds=balanced_schedule(jobs,speeds)
    maximum=[with_run(base,name,'B' if name=='B' else 'C',0. if name=='B' else 1.,0.,
        'confirmation',seed,1500) for seed in [101,202,303] for name in ['B','C','C0_matched']]
    _,main_seconds=balanced_schedule(maximum,speeds)
    projected=1.12*(dev_seconds+main_seconds)+1500
    fits=projected<=remaining(root)
    decision={'status':'BUDGET_FROZEN_BEFORE_VALIDATION' if fits else 'NOT_STARTED_remaining_grid_and_confirmation',
        'main_steps':1500,'fixed_attempts_before_any_validation':True,
        'forecast':{'development_makespan_seconds':dev_seconds,'maximum_confirmation_makespan_seconds':main_seconds,
            'multiplier':1.12,'evaluation_export_reserve_seconds':1500,'projected_seconds':projected},
        'remaining_seconds':remaining(root),'speeds':speeds,'validation_or_test_quality_seen':False,'utc':utc()}
    write(root/'budget_frozen_before_validation.json',decision)
    require(fits,'Whole fixed1500 cohort does not fit; NO automatic fidelity/grid/seed change')
    records=run_parallel(root,jobs,speeds,reserve=1500)
    require(source_map(root)==sources and aux_map(root)==aux,'Sources changed during development')
    write(root/'development_summary.json',{'status':'complete','study_plan_sha256':sha(root/'study_plan_frozen.json'),
        'budget_sha256':sha(root/'budget_frozen_before_validation.json'),
        'runs':[{'run_id':r['config']['run_id'],'complete_sha256':sha(root/'development'/r['config']['run_id']/'complete.json')}
            for r in [b]+records],'planned_runs':5,'complete_runs':5,
        'validation_quality_seen_by_launcher':False,'next_phase':'validation_only_selection_then_confirmation'})

def confirmation(root):
    plan=read(root/'study_plan_frozen.json');require(source_map(root)==plan['training_sources']
        and aux_map(root)==plan['auxiliary_sources'],'Frozen source changed')
    development_done=read(root/'development_summary.json')
    require(development_done['status']=='complete' and development_done['complete_runs']==5,'All5 development runs required')
    budget=read(root/'budget_frozen_before_validation.json')
    selection_path=root/'selection_frozen.json';selection=read(selection_path)
    require(selection['status']=='frozen_validation_only_selection_before_confirmation_and_test'
        and selection['study_plan_sha256']==sha(root/'study_plan_frozen.json')
        and selection['budget_sha256']==sha(root/'budget_frozen_before_validation.json')
        and selection['development_summary_sha256']==sha(root/'development_summary.json')
        and selection['test_quality_seen'] is False,'Prospective validation-only decision binding required')
    approval=read(root/'confirmation_sources_approved.json')
    require(approval['status']=='approved_frozen_v3_selection_before_confirmation_and_test'
        and approval['selection_sha256']==sha(selection_path) and approval['launcher_sha256']==sha(__file__)
        and approval['ai_root_approval'] is True,'Explicit confirmation approval required')
    ready,ready_sha,counts=data_check(root)
    require(ready_sha==plan['data_ready_sha256'],'Clean split changed')
    base=teacher_binding(root,common(root,ready,ready_sha,counts,plan['training_sources']))
    base['selection_sha256']=sha(selection_path)
    selected=selection['selected'];require(set(selected)=={'B','C'},'Exactly two selected methods required')
    for arm,value in selected.items():
        require(value['ema_decay'] in [.99,.999],'Selected EMA not in grid')
        require(value['coefficient'] in ([0.] if arm=='B' else [.1,1.] if arm in ['P','C'] else [.1,.5]),'Selected coefficient not in grid')
        require(value['kappa'] in ([0.,1.] if arm=='C' else [0.]),'Selected kappa not in grid')
    specs=[(arm,arm,selected[arm]['coefficient'],selected[arm]['kappa'],selected[arm]['ema_decay'])
        for arm in ['B','C']]
    if selected['C']['kappa']==1.:
        v=selected['C'];specs.append(('C0_matched','C',v['coefficient'],0.,v['ema_decay']))
    jobs=[]
    for seed in [101,202,303]:
        for name,arm,coefficient,kappa,ema in specs:
            jobs.append(with_run({**base,'selected_ema_decay':ema},name,arm,coefficient,kappa,
                'confirmation',seed,budget['main_steps']))
    require(not (root/'confirmation_summary.json').exists() and not (root/'confirmation').exists(),
        'Preserve prior confirmation; no retry')
    speeds=budget['speeds'];_,seconds=balanced_schedule(jobs,speeds)
    write(root/'confirmation_plan_frozen.json',{'status':'frozen_before_confirmation_and_test',
        'selection_sha256':sha(selection_path),'jobs':jobs,'maximum_seconds_estimate':1.12*seconds+1200,
        'remaining_seconds':remaining(root),'family_size':2,'conservative_reserved_slots':4,'one_sided_percentile_level':.9875})
    require(1.12*seconds+1200<=remaining(root),'Whole confirmation/test/export forecast no longer fits')
    records=run_parallel(root,jobs,speeds,reserve=1200)
    write(root/'confirmation_summary.json',{'status':'complete','study_plan_sha256':sha(root/'study_plan_frozen.json'),
        'selection_sha256':sha(selection_path),'confirmation_plan_sha256':sha(root/'confirmation_plan_frozen.json'),
        'planned_runs':len(jobs),'complete_runs':len(records),
        'runs':[{'run_id':r['config']['run_id'],'arm':r['arm'],'seed':r['seed'],
            'complete_sha256':sha(root/'confirmation'/r['config']['run_id']/'complete.json'),
            'checkpoint_sha256':r['checkpoint_sha256'],
            'durable_complete_sha256':r['durable_complete_sha256'],
            'selected_ema_decay':r['config']['selected_ema_decay']} for r in records],
        'test_quality_seen_by_launcher':False})

def main():
    require(sys.flags.optimize==0,'Optimized mode is not allowed for source/fixture gates')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=['development','confirmation'],required=True)
    args=parser.parse_args();root=args.root
    require(not root.is_symlink() and root.resolve()==ROOT,'Exact approved v3 root required')
    def terminate(*_):
        _STOP.set();raise KeyboardInterrupt('Owned v3 launcher termination requested')
    signal.signal(signal.SIGTERM,terminate)
    (development if args.phase=='development' else confirmation)(root)

if __name__=='__main__':main()
