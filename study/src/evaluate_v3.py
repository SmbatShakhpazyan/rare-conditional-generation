"""NEW validation-only HPO selection and gated fresh-v3 test evaluator.
No model, data, network or quality work on import. Frozen CleanFID/KID/PRDC
adapters are imported only after their literal hashes and approvals pass.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/kaggle/working/generative_recovery_2500')
STUDY='generative_recovery_1500'
PRIOR_READY_SHA='62347bb0d64d5d8fb816b8c233cdda41be30f89bb92d0a86bc30264f5f3ee639'
CLEAN_INDEX_SHA='d278e0a67060e70416314a05cb8ea29d6ccd41b80e22ab90c89226bd847156e1'
COMPONENTS_SHA='bb039432c9bc102fb7d0555a8d381293671d46c46a62bf624dcc0aab31239e8d'
WEIGHTS_SHA='f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4'
FROZEN={'evaluate.py':'29199ddb1aa44b972a9a0f8dcf28d96dbd4bd042f679895ac74c1b5d811ebb4f',
 'clean_features.py':'727e265b430a9276ef7ec36026d68d606970611c78fb3f9c582780fe69e488af',
 'diversity.py':'40ebbb953c800ae8c4287ed1b03b9f499388fde8760763a8874e182d4f005658',
 'models.py':'f82bfdd31060fe357a3102fa4debd110da5989f86c5f9e61b6163036f579f5fa'}
DEV=[('B','B',0.,0.),('C_l0p1_k0','C',.1,0.),('C_l0p1_k1','C',.1,1.),
 ('C_l1_k0','C',1.,0.),('C_l1_k1','C',1.,1.)]


def require(value,message):
    if not value:raise ValueError(message)


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):result.update(block)
    return result.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def safe(root,value):
    value=Path(value)
    if not value.is_absolute():value=root/value
    require('..' not in value.parts and not value.is_symlink(),'unsafe_owned_path')
    value=value.resolve();require(value.is_relative_to(root),'outside_owned_root')
    return value


def expected_dev():
    return {name+'_seed909':(arm,coefficient,kappa) for name,arm,coefficient,kappa in DEV}


def select_validation_points(points):
    """Pure metadata selector. Negative KIDs are retained; all10 are mandatory."""
    expected={(name,ema) for name in expected_dev() for ema in [.99,.999]}
    actual={(point['run_id'],point['ema_decay']) for point in points}
    require(len(points)==10 and actual==expected,'all10_distinct_declared_validation_points_required')
    records={arm:[] for arm in ['B','C']}
    for point in points:
        arm,coefficient,kappa=expected_dev()[point['run_id']]
        require(point['phase']=='validation' and point['arm']==arm
            and point['coefficient']==coefficient and point['kappa']==kappa
            and point['seed']==909 and point['step']==1500,'undeclared_validation_point')
        require(set(point['kid']['groups'])=={'0','1','2','3'},'all4_group_KIDs_required')
        scores=[point['kid']['groups'][str(g)]['kid'] for g in range(4)]
        require(all(isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) for x in scores),
            'nonfinite_or_missing_validation_score')
        value=float(scores[3]);tie=(round(value,12),float(coefficient),float(kappa),
            0 if point['ema_decay']==.999 else 1,point['run_id'])
        records[arm].append((tie,point,value))
    selected={};details={}
    for arm,values in records.items():
        tie,point,value=min(values,key=lambda x:x[0])
        selected[arm]={'coefficient':point['coefficient'],'kappa':point['kappa'],'ema_decay':point['ema_decay']}
        details[arm]={'run_id':point['run_id'],'raw_rare_validation_KID':value,'tie_key':list(tie),
            'EMA_sha256':point['EMA_sha256'],'complete_sha256':point['complete_sha256']}
    return selected,details


def uncertainty_status(phase,reports):
    if phase=='validation':
        return {'uncertainty_status':'NOT_PLANNED_FOR_VALIDATION_SELECTION',
            'all_planned_bootstrap_computed':None,'complete_scientific_report':False}
    computed=bool(reports) and all(value['status']=='computed' for value in reports.values())
    return {'uncertainty_status':'ALL_COMPUTED' if computed else 'INCOMPLETE',
        'all_planned_bootstrap_computed':computed,'complete_scientific_report':computed}


class Artifacts:
    def __init__(self,root):self.root=root;self.records={}
    def add(self,path):
        path=Path(path);relative=str(path.relative_to(self.root));digest=sha(path)
        require(relative not in self.records,'artifact_would_be_recommitted')
        self.records[relative]=digest
        with (self.root/'committed_artifacts.jsonl').open('a') as f:
            f.write(json.dumps({'path':relative,'sha256':digest,'bytes':path.stat().st_size})+'\n');f.flush();os.fsync(f.fileno())
        return digest


def gate(args):
    require(args.root==ROOT and ROOT.resolve()==ROOT and not ROOT.is_symlink(),'exact_v3_root_required')
    approval_path=ROOT/'evaluation_sources_approved.json';approval=read(approval_path)
    require(approval['status']=='approved_before_v3_validation_or_test_quality'
        and approval['study_id']==STUDY and approval['test_quality_seen'] is False,'prospective_quality_approval_required')
    stamp=datetime.fromisoformat(approval['approved_utc'].replace('Z','+00:00'))
    require(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),'approval_timestamp_invalid')
    for name,digest in approval['sources'].items():
        require(Path(name).name==name and sha(ROOT/'src'/name)==digest,'approved_evaluation_source_changed')
    require(approval['sources']['evaluate_v3.py']==sha(__file__)
        and 'component_bootstrap.py' in approval['sources'],'own_source_and_bootstrap_approval_required')
    for name,digest in FROZEN.items():require(sha(ROOT/'src'/name)==digest,'frozen_metric_or_architecture_changed')
    plan_path=ROOT/'study_plan_frozen.json';plan=read(plan_path)
    require(sha(plan_path)==approval['study_plan_sha256'] and plan['study_id']==STUDY
        and plan['status']=='frozen_before_teacher_grid_validation_test'
        and plan['development_steps']==1500
        and plan['selection']=='validation_group11_KID_min_rounded12_then_smallerlambda_then_kappa0_then_ema.999',
        'wrong_frozen_study_plan')
    require(plan['evaluation']['generated_per_group']==512 and plan['evaluation']['analysis_per_group']==512
        and plan['evaluation']['real_cap_per_group']==512
        and plan['evaluation']['generated_latent_coupling']=='shared_across_four_conditions_and_models'
        and plan['evaluation']['validation_z_seed']==64001 and plan['evaluation']['test_z_seed']==94001
        and plan['evaluation']['validation_real_hash_seed']==65001 and plan['evaluation']['test_real_hash_seed']==95001
        and plan['evaluation']['family_size']==2
        and plan['evaluation']['one_sided_percentile_level']==.9875
        and plan['evaluation']['bootstrap_replicates']==2000 and plan['evaluation']['bootstrap_seed']==95003,
        'frozen_evaluation_definition_changed')
    for name,digest in plan['training_sources'].items():require(sha(ROOT/'src'/name)==digest,'training_source_changed')
    for name,digest in plan['auxiliary_sources'].items():require(sha(safe(ROOT,name))==digest,'auxiliary_source_changed')
    ready_path=ROOT/'data/data_ready.json';ready=read(ready_path)
    require(sha(ready_path)==plan['data_ready_sha256'] and ready['status']=='frozen_clean_component_split'
        and ready['test_quality_accessed'] is False and ready['study_id']=='generative_recovery_2500'
        and ready['prior_definition_ready_sha256']==PRIOR_READY_SHA
        and ready['reconstructed_inputs_only'] is True,'clean_split_not_frozen')
    require(sha(ROOT/'data/clean_index.npz')==ready['clean_index_sha256']==CLEAN_INDEX_SHA
        and sha(ROOT/'data/component_ids.npy')==ready['component_ids_sha256']==COMPONENTS_SHA,'clean_input_array_changed')
    cache=Path(ready['cache_path'])
    require(not cache.is_symlink() and cache.is_relative_to(ROOT)
        and sha(cache)==ready['original_cache_sha256']=='98d39e0febb8c1946aba70eca2ca5cb96a657cdf6075f7ae32fc56d4c6f9bd5c',
        'original_cache_changed')
    budget_path=ROOT/'budget_frozen_before_validation.json';budget=read(budget_path)
    require(budget['status']=='BUDGET_FROZEN_BEFORE_VALIDATION' and budget['main_steps']==1500
        and budget['validation_or_test_quality_seen'] is False,'main_budget_not_frozen_before_validation')
    teacher=read(ROOT/'teacher/complete.json')
    require(teacher['status']=='complete' and teacher['source_plan_sha256']==sha(plan_path)
        and sha(ROOT/'teacher/fixed_teacher.pt')==teacher['teacher_artifact_sha256']
        and sha(ROOT/'teacher/statistics.json')==teacher['statistics_sha256'],'fixed_teacher_binding_changed')
    require(sha(args.weights_dir/'inception-2015-12-05.pt')==WEIGHTS_SHA,'actual_CleanFID_weights_changed')
    return approval,plan,ready,budget,cache


def completed_runs(phase,plan,budget,ready):
    confirmation_jobs=None
    if phase=='validation':
        require(not (ROOT/'confirmation').exists() and not (ROOT/'test_evaluation').exists(),
            'validation_must_precede_confirmation_and_new_test')
        summary_path=ROOT/'development_summary.json';summary=read(summary_path)
        require(summary['status']=='complete' and summary['planned_runs']==summary['complete_runs']==5
            and summary['study_plan_sha256']==sha(ROOT/'study_plan_frozen.json')
            and summary['budget_sha256']==sha(ROOT/'budget_frozen_before_validation.json'),'full5_development_required')
        entries=summary['runs'];require({x['run_id'] for x in entries}==set(expected_dev()),'wrong_development_grid')
        phase_folder='development';selection=None
    else:
        selection=read(ROOT/'selection_frozen.json')
        require(set(selection['selected'])=={'B','C'},'only_fresh_B_C_selection_allowed')
        require(selection['status']=='frozen_validation_only_selection_before_confirmation_and_test'
            and selection['study_plan_sha256']==sha(ROOT/'study_plan_frozen.json')
            and selection['test_quality_seen'] is False,'validation_selection_not_locked')
        validation=ROOT/'validation_evaluation'
        validation_index=read(validation/'results_index.json')
        registry=read(validation/'artifact_registry.json')['sha256']
        require(validation_index['status']=='COMPLETED' and validation_index['point_count']==10
            and sha(validation/'results_index.json')==selection['validation_results_index_sha256'],
            'complete_validation_index_required')
        validation_points=[]
        for relative in validation_index['points'].values():
            point_path=safe(validation,relative)
            require(sha(point_path)==registry[relative],'validation_point_not_committed_or_changed')
            validation_points.append(read(point_path))
        recomputed,_=select_validation_points(validation_points)
        require(recomputed==selection['selected'],'frozen_selection_differs_from_all10_validation_points')
        selection_approval=read(ROOT/'confirmation_sources_approved.json')
        require(selection_approval['selection_sha256']==sha(ROOT/'selection_frozen.json')
            and selection_approval['status']=='approved_frozen_v3_selection_before_confirmation_and_test'
            and selection_approval['ai_root_approval'] is True,'confirmation_approval_required')
        summary_path=ROOT/'confirmation_summary.json';summary=read(summary_path)
        confirmation=read(ROOT/'confirmation_plan_frozen.json')
        require(summary['status']=='complete' and summary['planned_runs']==summary['complete_runs']
            and summary['selection_sha256']==sha(ROOT/'selection_frozen.json')
            and summary['confirmation_plan_sha256']==sha(ROOT/'confirmation_plan_frozen.json')
            and len(confirmation['jobs'])==summary['complete_runs'],'all_planned_confirmation_required')
        confirmation_jobs={job['run_id']:job for job in confirmation['jobs']}
        expected_main={name+'_seed'+str(seed) for name in ['B','C']+
            (['C0_matched'] if selection['selected']['C']['kappa']==1. else []) for seed in [101,202,303]}
        require(set(confirmation_jobs)==expected_main,'exact_B_C_and_conditional_C0_fresh_three_seed_plan_required')
        require(len(confirmation_jobs)==len(confirmation['jobs']),'duplicate_confirmation_job')
        entries=summary['runs'];require({x['run_id'] for x in entries}==set(confirmation_jobs),
            'confirmation_run_set_differs')
        phase_folder='confirmation'
    records=[]
    for entry in entries:
        run_id=entry['run_id'];require(Path(run_id).name==run_id,'unsafe_run_ID')
        folder=ROOT/phase_folder/run_id;complete_path=folder/'complete.json';complete=read(complete_path);config=complete['config']
        require(sha(complete_path)==entry['complete_sha256'] and complete['status']=='complete'
            and complete['config']['source_sha256']==plan['training_sources']
            and complete['arm']==config['arm'] and complete['seed']==config['seed']
            and config['run_id']==run_id and config['phase']==phase_folder
            and config['data_ready_sha256']==plan['data_ready_sha256']
            and config['metadata_npz']==str(ROOT/'data/clean_index.npz')
            and config['index_sha256']==ready['clean_index_sha256']
            and config['component_ids_sha256']==ready['component_ids_sha256']
            and complete['completed_steps']==config['steps']==(1500 if phase=='validation' else budget['main_steps'])
            and complete['real_images']==config['steps']*64 and complete['group_exposures']==[config['steps']*16]*4,
            'completed_training_binding_changed')
        if confirmation_jobs is not None:
            require(config==confirmation_jobs[run_id],'confirmation_config_differs_from_exact_frozen_job')
        if 'teacher_artifact_sha256' in config or config['arm']=='C':
            require(config['teacher_artifact_path']==str(ROOT/'teacher/fixed_teacher.pt')
                and config['teacher_statistics_path']==str(ROOT/'teacher/statistics.json')
                and config['teacher_artifact_sha256']==sha(ROOT/'teacher/fixed_teacher.pt')
                and config['teacher_statistics_sha256']==sha(ROOT/'teacher/statistics.json'),
                'run_teacher_or_statistics_binding_changed')
        require(sha(folder/'latest.pt')==complete['checkpoint_sha256'],'source_checkpoint_changed')
        import numpy as np
        exposure=np.load(folder/'exposures.npy',allow_pickle=False)
        with np.load(ROOT/'data/clean_index.npz',allow_pickle=False) as values:clean_splits=values['splits']
        require(exposure.shape==(202599,) and exposure.dtype==np.int64 and bool((exposure>=0).all())
            and int(exposure.sum())==config['steps']*64 and not bool((exposure[clean_splits!=0]>0).any()),
            'actual_training_exposures_include_heldout_or_differ')
        if phase=='validation':
            arm,coefficient,kappa=expected_dev()[run_id]
            require(config['arm']==arm and config['coefficient']==coefficient and config['kappa']==kappa
                and config['seed']==909,'undeclared_development_config')
            decays=[.99,.999]
        else:
            name=run_id.rsplit('_seed',1)[0];method='C' if name=='C0_matched' else name
            require(method in selection['selected'] and config['seed'] in [101,202,303]
                and config['selection_sha256']==sha(ROOT/'selection_frozen.json'),'unselected_confirmation_config')
            chosen=selection['selected'][method]
            require(config['coefficient']==chosen['coefficient'] and config['kappa']==(0. if name=='C0_matched' else chosen['kappa'])
                and config['selected_ema_decay']==chosen['ema_decay'],'confirmation_hyperparameter_changed')
            decays=[chosen['ema_decay']]
        for decay in decays:
            item=complete['ema_artifacts'][str(decay)];ema=safe(ROOT,item['path'])
            require(ema==folder/('EMA_'+str(decay)+'.pt') and sha(ema)==item['sha256'],'EMA_artifact_changed')
            records.append({'run_id':run_id,'arm':config['arm'],'coefficient':config['coefficient'],'kappa':config['kappa'],
                'seed':config['seed'],'step':config['steps'],'ema_decay':decay,'EMA_path':str(ema),'EMA_sha256':item['sha256'],
                'complete_sha256':entry['complete_sha256'],'config':config})
    return records,summary_path


def shared_latents(seed,count=512,dimension=128):
    import numpy as np
    return np.random.default_rng(seed).standard_normal((count,dimension),dtype=np.float32)


def real_rows(names,groups,splits,split,seed):
    import numpy as np
    selected=[]
    for group in range(4):
        rows=np.flatnonzero((splits==split)&(groups==group)).tolist()
        rows=sorted(rows,key=lambda row:(hashlib.sha256(f'{seed}|{group}|{names[row]}'.encode()).digest(),str(names[row])))[:512]
        rows.sort(key=lambda row:str(names[row]));require(len(rows)>=100,'heldout_support_floor')
        selected.extend(rows)
    return np.asarray(selected,dtype=np.int64)


def save_features(path,dataset,artifacts):
    import numpy as np
    value=np.asarray(dataset['features']);compact=value.astype(np.float32)
    require(np.array_equal(compact.astype(np.float64),value),'extractor_features_not_lossless_float32')
    arrays={key:np.asarray(item) for key,item in dataset.items() if key!='provenance'};arrays['features']=compact
    np.savez_compressed(path,**arrays);digest=artifacts.add(path)
    sidecar=Path(str(path)+'.json');write(sidecar,{'npz_sha256':digest,'provenance':dataset['provenance'],
        'array_rows':len(value),'feature_dtype':'float32_lossless_from_extractor_float32','feature_dimension':2048})
    artifacts.add(sidecar);return digest


def worker(args):
    approval,plan,ready,budget,cache_path=gate(args)
    jobs,summary_path=completed_runs(args.phase,plan,budget,ready)
    output=ROOT/(args.phase+'_evaluation');require(not output.exists() and not output.is_symlink(),'fresh_evaluation_only_no_silent_resume')
    if args.phase=='validation':require(not (ROOT/'selection_frozen.json').exists(),'selection_already_exists')
    output.mkdir();artifacts=Artifacts(output);began=time.monotonic()
    binding={'status':'bound_before_any_v3_'+args.phase+'_quality','utc':utc(),'study_id':STUDY,
        'evaluation_source_sha256':sha(__file__),'approval_sha256':sha(ROOT/'evaluation_sources_approved.json'),
        'study_plan_sha256':sha(ROOT/'study_plan_frozen.json'),'budget_sha256':sha(ROOT/'budget_frozen_before_validation.json'),
        'data_ready_sha256':sha(ROOT/'data/data_ready.json'),'complete_summary_sha256':sha(summary_path),
        'phase':args.phase,'jobs':jobs,'latent_coupling':'shared_across_four_conditions_and_models',
        'generated_per_group':512,'real_cap_per_group':512,'analysis_cap':512,'feature_weights_sha256':WEIGHTS_SHA,
        'no_classifier_filter_or_relabeling':True,'test_quality_seen_before_validation_selection':False,
        'input_files_sha256':{str(path):sha(path) for path in [ROOT/'study_plan_frozen.json',
            ROOT/'budget_frozen_before_validation.json',ROOT/'data/data_ready.json',ROOT/'data/clean_index.npz',
            ROOT/'data/component_ids.npy',summary_path,ROOT/'teacher/complete.json',ROOT/'teacher/statistics.json',
            ROOT/'teacher/fixed_teacher.pt',args.weights_dir/'inception-2015-12-05.pt']}}

    write(output/'binding_before_quality.json',binding);artifacts.add(output/'binding_before_quality.json')
    old=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('v3_evaluation_inner_relative_cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(1,args.max_seconds-5))
    try:
        import numpy as np
        import torch
        from PIL import Image
        # gate verified every frozen source BEFORE these imports.
        from clean_features import CleanFIDFeatures
        from models import Generator
        from evaluate import rgb_uint8,evaluate_feature_groups
        from diversity import evaluate_prdc_groups
        extractor=CleanFIDFeatures(device=args.device,weights_dir=str(args.weights_dir),allow_download=False,
            expected_weights_sha256=WEIGHTS_SHA)
        require(extractor.provenance['feature_dim']==2048,'feature_dimension_changed')
        with np.load(ROOT/'data/clean_index.npz',allow_pickle=False) as index:
            names=index['filenames'];groups=index['groups'];splits=index['splits']
        components=np.load(ROOT/'data/component_ids.npy',allow_pickle=False)
        sets=[set(components[splits==s].tolist()) for s in range(3)]
        require(all(not sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)]),'leakage_components_cross_splits')
        split=1 if args.phase=='validation' else 2
        seed=64001 if split==1 else 94001;real_seed=65001 if split==1 else 95001
        rows=real_rows(names,groups,splits,split,real_seed)
        cache=np.load(cache_path,mmap_mode='r',allow_pickle=False)
        feature_batches=[]
        for start in range(0,len(rows),64):
            image_batch=np.ascontiguousarray(cache[rows[start:start+64]].transpose(0,2,3,1))
            feature_batches.append(extractor(image_batch))
        provenance={'feature_extractor':extractor.provenance,'selection':'fixed_filename_hash_before_features',
            'split':args.phase,'data_ready_sha256':sha(ROOT/'data/data_ready.json')}
        real={'features':np.concatenate(feature_batches),'groups':groups[rows],
            'ids':names[rows].astype(str),'component_ids':components[rows],'original_rows':rows,'provenance':provenance}
        real_sha=save_features(output/'real_features.npz',real,artifacts)
        latents=shared_latents(seed);np.save(output/'shared_latents.npy',latents,allow_pickle=False)
        latent_sha=artifacts.add(output/'shared_latents.npy')
        all_points=[];fake_datasets={};points={}
        for job in jobs:
            require(sha(job['EMA_path'])==job['EMA_sha256'],'EMA_changed_immediately_before_safe_load')
            payload=torch.load(job['EMA_path'],map_location='cpu',weights_only=True)
            require(payload['study_id']==STUDY and payload['config']==job['config']
                and payload['step']==job['step'] and payload['ema_decay']==job['ema_decay']
                and payload['source_sha256']==plan['training_sources'],'safe_EMA_payload_binding_changed')
            gconfig=payload['generator_config']
            require(gconfig['z_dim']==128 and gconfig['num_classes']==4 and gconfig['base_channels']==32
                and gconfig['image_size']==64,'common_generator_architecture_changed')
            generator=Generator(z_dim=128,num_classes=4,base_channels=32).to(args.device)
            generator.load_state_dict(payload['ema_generator'],strict=True);generator.eval().requires_grad_(False)
            feature_batches=[];fake_groups=[];ids=[];latent_indices=[]
            key=job['run_id']+'_ema'+str(job['ema_decay'])
            for group in range(4):
                grid=[]
                for start in range(0,512,64):
                    with torch.inference_mode():
                        values=generator(torch.from_numpy(latents[start:start+64]).to(args.device),
                            torch.full((min(64,512-start),),group,dtype=torch.long,device=args.device))
                    images=values.detach().float().cpu().numpy().transpose(0,2,3,1)
                    pixels=rgb_uint8(images)
                    if start==0:grid=pixels[:16].copy()
                    feature_batches.append(extractor(pixels));fake_groups.extend([group]*len(pixels))
                    ids.extend([f'{args.phase}:z{seed}:g{group}:i{i:06d}' for i in range(start,start+len(pixels))])
                    latent_indices.extend(range(start,start+len(pixels)))
                canvas=np.zeros((256,256,3),dtype=np.uint8)
                for i,image in enumerate(grid):canvas[(i//4)*64:(i//4+1)*64,(i%4)*64:(i%4+1)*64]=image
                path=output/'generated_grids'/key/('group'+str(group)+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(canvas).save(path);artifacts.add(path)
            fake={'features':np.concatenate(feature_batches),'groups':np.asarray(fake_groups,dtype=np.int64),
                'ids':np.asarray(ids),'latent_indices':np.asarray(latent_indices,dtype=np.int64),
                'provenance':{'feature_extractor':extractor.provenance,'requested_bins':True,
                    'EMA_sha256':job['EMA_sha256'],'shared_latents_sha256':latent_sha}}
            path=output/'features'/('fake_'+key+'.npz');path.parent.mkdir(exist_ok=True)
            fake_sha=save_features(path,fake,artifacts)
            kid=evaluate_feature_groups(real,fake,analysis_cap=512,selection_seed=real_seed)
            prdc=evaluate_prdc_groups(real,fake,nearest_k=3,analysis_cap=512,selection_seed=real_seed,kid_point_report=kid)
            point={**job,'phase':args.phase,'kid':kid,'prdc':prdc,'feature_definition':extractor.provenance,
                'real_feature_sha256':real_sha,'fake_feature_sha256':fake_sha,'shared_latents_sha256':latent_sha,
                'clean_index_sha256':ready['clean_index_sha256'],'component_ids_sha256':ready['component_ids_sha256'],
                'study_plan_sha256':binding['study_plan_sha256'],'teacher_statistics_sha256':sha(ROOT/'teacher/statistics.json'),
                'no_generated_filtering_or_relabeling':True}
            point_path=output/'points'/(key+'.json');write(point_path,point);artifacts.add(point_path)
            all_points.append(point);points[key]=str(point_path.relative_to(output));fake_datasets[key]=fake
            del generator,payload;torch.cuda.empty_cache()
            print(json.dumps({'phase':args.phase,'completed_points':len(points),'planned_points':len(jobs),'key':key}),flush=True)
        point_lookup={(p['run_id'],p['ema_decay']):p for p in all_points}
        bootstrap_reports={}
        if args.phase=='test':
            from component_bootstrap import component_paired_bootstrap
            selection=read(ROOT/'selection_frozen.json')
            def models(name):
                method='C' if name=='C0_matched' else name;decay=selection['selected'][method]['ema_decay']
                return [fake_datasets[f'{name}_seed{s}_ema{decay}'] for s in [101,202,303]]
            def ema_hashes(name):
                method='C' if name=='C0_matched' else name;decay=selection['selected'][method]['ema_decay']
                return [point_lookup[(f'{name}_seed{s}',decay)]['EMA_sha256'] for s in [101,202,303]]
            refs=['B']+(['C0_matched'] if selection['selected']['C']['kappa']==1. else [])
            for reference in refs:
                report=component_paired_bootstrap(real,models('C'),models(reference),replicates=2000,
                    seed=95003,confidence=.9875,max_seconds=min(600,max(1,args.max_seconds-(time.monotonic()-began)-10)))
                report['comparison']={'candidate':'C','reference':reference,'fixed_training_seeds':[101,202,303],
                    'candidate_EMA_sha256':ema_hashes('C'),
                    'reference_EMA_sha256':ema_hashes(reference)}
                report['real_feature_sha256']=real_sha;report['feature_definition']=extractor.provenance
                path=output/'bootstrap'/('C_vs_'+reference+'.json');write(path,report);artifacts.add(path)
                bootstrap_reports[reference]={'path':str(path.relative_to(output)),'status':report['status']}
        require(all(sha(ROOT/'src'/name)==digest for name,digest in approval['sources'].items()),'evaluation_sources_changed')
        require(all(sha(path)==digest for path,digest in binding['input_files_sha256'].items()),'evaluation_inputs_changed')
        require(all(sha(job['EMA_path'])==job['EMA_sha256'] for job in jobs),'evaluated_EMA_files_changed')
        index={'status':'COMPLETED','study_id':STUDY,'phase':args.phase,'points':points,'point_count':len(points),
            'bootstrap':bootstrap_reports,**uncertainty_status(args.phase,bootstrap_reports),
            'all_planned_point_metrics_computed':True,'binding_sha256':sha(output/'binding_before_quality.json'),
            'elapsed_seconds':time.monotonic()-began,'no_semantic_or_training_population_guarantee':True}
        write(output/'results_index.json',index);artifacts.add(output/'results_index.json')
        if args.phase=='validation':
            selected,details=select_validation_points(all_points)
            selection={'status':'frozen_validation_only_selection_before_confirmation_and_test','utc':utc(),
                'study_id':STUDY,'study_plan_sha256':binding['study_plan_sha256'],
                'budget_sha256':binding['budget_sha256'],'development_summary_sha256':sha(summary_path),
                'validation_results_index_sha256':sha(output/'results_index.json'),
                'evaluation_binding_sha256':sha(output/'binding_before_quality.json'),
                'data_ready_sha256':sha(ROOT/'data/data_ready.json'),'teacher_statistics_sha256':sha(ROOT/'teacher/statistics.json'),
                'clean_index_sha256':ready['clean_index_sha256'],'component_ids_sha256':ready['component_ids_sha256'],
                'selected':selected,'selection_details':details,'test_quality_seen':False,
                'criterion':'only_raw_rare_group11_validation_KID_round12_then_coefficient_then_kappa_thenEMA999',
                'all10_tried_points':[{k:p[k] for k in ['run_id','ema_decay','EMA_sha256','complete_sha256']}
                    |{'raw_rare_validation_KID':p['kid']['groups']['3']['kid']} for p in all_points],
                'fixed_main_steps_chosen_before_validation':budget['main_steps'],
                'matched_kappa0_status':'PLANNED' if selected['C']['kappa']==1. else 'NOT_APPLICABLE_SELECTED_KAPPA_ZERO',
                'original_test_quality_accessed_for_v3_design_or_selection':False,
                'selection_attestations_are_operator_provenance_not_a_cryptographic_proof_of_unseen_scores':True}
            write(ROOT/'selection_frozen.json',selection)
        write(output/'artifact_registry.json',{'status':'committed','sha256':artifacts.records})
        print(json.dumps({'status':'COMPLETED','phase':args.phase,'points':len(points)}),flush=True)
    except BaseException as error:
        write(output/'failure.json',{'status':'INCOMPLETE','error_type':type(error).__name__,
            'partial_records_preserved':True,'no_automatic_retry':True,'elapsed_seconds':time.monotonic()-began})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def coordinator(args):
    gate(args)
    deadline=datetime.fromisoformat(read(ROOT/'study_clock.json')['deadline_utc'].replace('Z','+00:00'))
    remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
    require(remaining>=30,'authorized_study_deadline_has_no_evaluation_budget')
    args.max_seconds=min(args.max_seconds,int(remaining))
    output=ROOT/(args.phase+'_evaluation');guard=ROOT/(args.phase+'_evaluation_guard')
    require(not output.exists() and not guard.exists() and not output.is_symlink() and not guard.is_symlink(),
        'fresh_evaluation_and_guard_required')
    guard.mkdir();began=time.monotonic();process=None;code=None;timed_out=False
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--phase',args.phase,
        '--root',str(ROOT),'--device',args.device,'--weights-dir',str(args.weights_dir),'--max-seconds',str(args.max_seconds)]
    try:
        with (guard/'stdout.log').open('xb') as stdout,(guard/'stderr.log').open('xb') as stderr:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
            try:code=process.wait(timeout=args.max_seconds)
            except subprocess.TimeoutExpired:timed_out=True
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                code=process.returncode
    finally:
        status={'status':'COMPLETED' if code==0 and not timed_out else 'INCOMPLETE','return_code':code,
            'timed_out':timed_out,'elapsed_seconds':time.monotonic()-began,'cap_seconds':args.max_seconds,
            'source_sha256':sha(__file__),'phase':args.phase,'partials_preserved':True,'no_automatic_retry':True}
        write(guard/'process.json',status)
    print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--phase',choices=['validation','test'],required=True)
    parser.add_argument('--weights-dir',type=Path,default=Path('/tmp'));parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-seconds',type=int,default=1200);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();require(sys.flags.optimize==0 and 30<=args.max_seconds<=1800,'invalid_runtime_settings')
    if args.worker:worker(args)
    else:coordinator(args)
