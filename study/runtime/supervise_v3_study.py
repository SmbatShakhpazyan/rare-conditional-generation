"""One source-approved V3 chain; algorithmic validation selection only.

No work/model imports on import. Fixed1500, fiveDEV/tenVAL, fresh6/9MAIN. The
root's prospective approval authorizes derived source approvals, not human acts.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/kaggle/working/generative_recovery_2500')
DEADLINE=datetime(2026,11,19,11,53,tzinfo=timezone.utc)

def require(value,message):
    if not value:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def utc():return datetime.now(timezone.utc).isoformat()
def remaining():return (DEADLINE-datetime.now(timezone.utc)).total_seconds()

def stop_known(process):
    if process is None or process.poll() is not None:return
    # Each reviewed coordinator owns separate worker sessions and cleans them
    # on KeyboardInterrupt. Give that bounded cleanup time before escalation.
    for sig,grace in [(signal.SIGINT,30),(signal.SIGTERM,5),(signal.SIGKILL,5)]:
        if process.poll() is not None:break
        try:os.killpg(process.pid,sig)
        except ProcessLookupError:pass
        try:process.wait(timeout=grace)
        except subprocess.TimeoutExpired:continue
    require(process.poll() is not None,'Owned coordinator cleanup incomplete')

def run(root,name,command,cap):
    require(cap>=1,'No authorized phase budget remains')
    folder=root/'runtime/supervisor';process=None;started=time.monotonic();code=None;timed_out=False
    try:
        with (folder/(name+'.log')).open('xb') as log:
            process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'PYTHONUNBUFFERED':'1','CUDA_VISIBLE_DEVICES':'0,1',
                     'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            write(folder/(name+'.started.json'),{'status':'OWNED_PHASE_STARTED','phase':name,'pid':process.pid,
                'command':command,'started_utc':utc(),'cap_seconds':cap,'absolute_deadline_utc':DEADLINE.isoformat()})
            while process.poll() is None:
                if time.monotonic()-started>=cap or remaining()<=0:
                    timed_out=True;stop_known(process);break
                time.sleep(.5)
            code=process.wait(timeout=5)
    finally:
        stop_known(process)
        write(folder/(name+'.process.json'),{'status':'COMPLETED' if code==0 and not timed_out else 'INCOMPLETE',
            'return_code':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-started,'phase':name,
            'child_pid':None if process is None else process.pid,'ended_utc':utc(),'no_automatic_retry':True})
    require(code==0 and not timed_out,'Phase failed; preserve actual records, no retry: '+name)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.root
    require(sys.flags.optimize==0 and root==ROOT and root.resolve()==ROOT and not root.is_symlink(),'Exact V3 root/nonoptimized mode required')
    approved_path=root/'supervisor_sources_approved.json';approved=read(approved_path)
    require(approved['status']=='approved_prospective_v3_all_phases_before_any_quality'
        and approved['study_id']=='generative_recovery_1500' and approved['supervisor_sha256']==sha(__file__)
        and approved['deadline_utc']=='2026-11-19T11:53:00Z' and approved['main_steps']==1500
        and approved['validation_points']==10 and approved['ai_root_approval'] is True
        and approved['algorithmic_selection_and_derived_source_approvals_authorized'] is True,
        'Prospective root all-phase/source/budget authorization required')
    for name,digest in approved['sources'].items():
        path=root/name;require(path.is_file() and not path.is_symlink() and sha(path)==digest,'Approved source differs: '+name)
    require(approved['data_preparation_source_path']=='runtime/prepare_recovery_zip_data.py'
        and approved['data_preparation_source_sha256']=='40b140d0c0675571b94b23ac07d7cf36f817f0ffeae54c5359678530248e6fee',
        'Only new source-approved ZIP recovery may unlock this model study')
    prep=root/approved['data_preparation_source_path']
    require(sha(prep)==approved['data_preparation_source_sha256'],'Preparation source differs')
    require(remaining()>30,'Absolute recovery deadline elapsed')
    folder=root/'runtime/supervisor';require(not folder.exists() and not folder.is_symlink(),'Preserve prior supervisor attempt')
    folder.mkdir();write(folder/'started.json',{'status':'V3_CHAIN_STARTED','utc':utc(),'approval_sha256':sha(approved_path),
        'supervisor_sha256':sha(__file__),'absolute_deadline_utc':DEADLINE.isoformat(),'main_steps':1500,
        'operator_quality_numbers_printed':False,'prior_V2_model_or_score_pooling':False})
    try:
        print('V3_WAITING_FOR_EXACT_APPROVED_DATA_READY; fixed1500 before any quality',flush=True)
        ready_path=root/'data/data_ready.json';prep_process_path=root/'zip_preparation_guard/process.json'
        while True:
            require(remaining()>1200,'No sufficient authorized recovery time remains')
            try:prep_process=read(prep_process_path)
            except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
            require(prep_process['status']=='COMPLETED' and prep_process['return_code']==0
                and prep_process['source_sha256']==approved['data_preparation_source_sha256'],
                'New ZIP preparation did not complete; preserve failure and do not train')
            require(ready_path.is_file() and prep_process['ready_sha256']==sha(ready_path),'Terminal preparation/ready binding differs')
            break
        ready=read(ready_path)
        expected={'original_cache_sha256':'98d39e0febb8c1946aba70eca2ca5cb96a657cdf6075f7ae32fc56d4c6f9bd5c',
            'clean_index_sha256':'d278e0a67060e70416314a05cb8ea29d6ccd41b80e22ab90c89226bd847156e1',
            'component_ids_sha256':'bb039432c9bc102fb7d0555a8d381293671d46c46a62bf624dcc0aab31239e8d'}
        require(ready['status']=='frozen_clean_component_split' and ready['study_id']=='generative_recovery_2500'
            and ready['reconstructed_inputs_only'] is True and ready['test_quality_accessed'] is False
            and ready['original_val_test_images_used'] is False
            and ready['prior_definition_ready_sha256']=='62347bb0d64d5d8fb816b8c233cdda41be30f89bb92d0a86bc30264f5f3ee639'
            and ready['preparation_source_path']==approved['data_preparation_source_path']
            and ready['source_sha256']==sha(prep)==approved['data_preparation_source_sha256']
            and all(ready[key]==value for key,value in expected.items()),'Actual exact reconstructed data definition required')
        for relative,key in [('data/images64.npy','original_cache_sha256'),('data/clean_index.npz','clean_index_sha256'),
                             ('data/component_ids.npy','component_ids_sha256')]:
            path=root/relative;require(path.is_file() and not path.is_symlink() and sha(path)==expected[key],
                'Actual cache/index/component bytes differ')
        for name,digest in approved['sources'].items():require(sha(root/name)==digest,'Source changed while awaiting data')
        training_names=['models.py','model_losses.py','compensated_interaction.py','train_v3.py',
                        'interaction_witness.py','interaction_statistics.py']
        aux_names=['src/prepare_teacher_v3.py','runtime/archive_v3_model.py','tests/test_interaction_statistics.py',
                   'tests/test_interaction_witness_torch.py','runtime/launch_v3_study.py']
        write(root/'training_sources_approved.json',{'status':'approved_before_v3_teacher_grid_validation_and_test',
            'study_id':'generative_recovery_1500','training_sources':{name:approved['sources']['src/'+name] for name in training_names},
            'auxiliary_sources':{name:approved['sources'][name] for name in aux_names},
            'data_ready_sha256':sha(ready_path),'deadline_utc':'2026-11-19T11:53:00Z','max_study_seconds':6000,
            'test_quality_seen':False,'ai_root_approval':True,'human_approval_claim':False,
            'main_attempts_fixed_before_any_validation':1500,'approved_utc':utc(),
            'authorized_by_prospective_supervisor_approval_sha256':sha(approved_path)})
        write(folder/'actual_ready_bound.json',{'data_ready_sha256':sha(ready_path),'preparation_source_sha256':sha(prep),
            'training_approval_sha256':sha(root/'training_sources_approved.json'),'utc':utc()})
        run(root,'development',[sys.executable,'-B',str(root/'runtime/launch_v3_study.py'),'--root',str(root),
            '--phase','development'],max(1,remaining()-1200))
        summary=read(root/'development_summary.json')
        require(summary['status']=='complete' and summary['complete_runs']==summary['planned_runs']==5,'AllfiveDEVrequired')
        plan=read(root/'study_plan_frozen.json')
        require(plan['development_steps']==1500 and read(root/'budget_frozen_before_validation.json')['main_steps']==1500,
            'Fixedbudget changed')
        eval_approval={'status':'approved_before_v3_validation_or_test_quality','study_id':'generative_recovery_1500',
            'test_quality_seen':False,'approved_utc':utc(),'ai_root_approval':True,'human_approval_claim':False,
            'authorized_by_prospective_supervisor_sha256':sha(approved_path),
            'study_plan_sha256':sha(root/'study_plan_frozen.json'),
            'sources':{Path(name).name:digest for name,digest in approved['sources'].items() if name.startswith('src/')}}
        require('evaluate_v3.py' in eval_approval['sources'] and 'component_bootstrap.py' in eval_approval['sources'],
            'Evaluation sources absent from prospective authorization')
        write(root/'evaluation_sources_approved.json',eval_approval)
        eval_command=[sys.executable,'-B',str(root/'src/evaluate_v3.py'),'--root',str(root),'--weights-dir','/tmp','--device','cuda:0']
        run(root,'validation',eval_command+['--phase','validation','--max-seconds',str(min(900,max(30,int(remaining())-10)))],
            min(960,max(1,remaining()-5)))
        index=read(root/'validation_evaluation/results_index.json');selection=read(root/'selection_frozen.json')
        require(index['status']=='COMPLETED' and index['point_count']==10 and set(selection['selected'])=={'B','C'}
            and selection['status']=='frozen_validation_only_selection_before_confirmation_and_test'
            and selection['test_quality_seen'] is False,'AlltenVALandfrozenchoice required')
        evaluator=root/'src/evaluate_v3.py'
        require(sha(evaluator)==approved['sources']['src/evaluate_v3.py'],'Pure selector source changed')
        spec=importlib.util.spec_from_file_location('_approved_v3_pure_selector',evaluator)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        registry=read(root/'validation_evaluation/artifact_registry.json')['sha256'];points=[]
        for relative in index['points'].values():
            path=root/'validation_evaluation'/relative
            require(sha(path)==registry[relative],'Actual committed VAL point changed');points.append(read(path))
        selected,_=module.select_validation_points(points)
        require(selected==selection['selected'],'Algorithmic selected parameters differ')
        write(root/'confirmation_sources_approved.json',{'status':'approved_frozen_v3_selection_before_confirmation_and_test',
            'selection_sha256':sha(root/'selection_frozen.json'),'launcher_sha256':sha(root/'runtime/launch_v3_study.py'),
            'ai_root_approval':True,'human_approval_claim':False,'approved_utc':utc(),
            'authorized_by_prospective_supervisor_sha256':sha(approved_path)})
        run(root,'confirmation',[sys.executable,'-B',str(root/'runtime/launch_v3_study.py'),'--root',str(root),
            '--phase','confirmation'],max(1,remaining()-900))
        main_summary=read(root/'confirmation_summary.json');expected=9 if selected['C']['kappa']==1. else 6
        require(main_summary['status']=='complete' and main_summary['complete_runs']==main_summary['planned_runs']==expected,
            'Allplannedfreshmainmodels requiredbeforetest')
        run(root,'test',eval_command+['--phase','test','--max-seconds',str(min(900,max(30,int(remaining())-10)))],
            min(960,max(1,remaining()-5)))
        result=read(root/'test_evaluation/results_index.json')
        write(folder/'complete.json',{'status':'CHAIN_POINT_PHASES_COMPLETED','utc':utc(),'main_models':expected,
            'test_points':result['point_count'],'uncertainty_status':result.get('uncertainty_status'),
            'complete_scientific_report':result.get('complete_scientific_report',False),
            'validation_selection_sha256':sha(root/'selection_frozen.json'),'no_test_based_HPO':True})
        print('V3_CHAIN_POINT_PHASES_COMPLETED; see explicit uncertainty status and durable model archives',flush=True)
    except BaseException as error:
        write(folder/'failure.json',{'status':'V3_CHAIN_INCOMPLETE_PRESERVED','error_type':type(error).__name__,
            'utc':utc(),'no_automatic_retry':True});raise

if __name__=='__main__':
    def terminate(*_):raise KeyboardInterrupt('Owned supervisor stop requested')
    signal.signal(signal.SIGTERM,terminate)
    main()
