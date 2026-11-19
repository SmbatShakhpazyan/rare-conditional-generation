"""Portable delivered-v2 MAIN EMA replay, never an original-producer resume.

Preflight is stdlib-only. Smoke computes one C101/condition0/64 feature batch,
no KID. Full mode regenerates all512 shared-z requests/group; CI is explicit.
No work/model/Torch/network on import; all output is a NEW finite-budget tree.
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

VERIFIER_SHA='307abd988ab2bdfc926b03fb4fed49735071d1c1c3465e05cbdc98df6c38eb9d'
FROZEN={'models.py':'f82bfdd31060fe357a3102fa4debd110da5989f86c5f9e61b6163036f579f5fa',
 'evaluate.py':'29199ddb1aa44b972a9a0f8dcf28d96dbd4bd042f679895ac74c1b5d811ebb4f',
 'clean_features.py':'727e265b430a9276ef7ec36026d68d606970611c78fb3f9c582780fe69e488af',
 'diversity.py':'40ebbb953c800ae8c4287ed1b03b9f499388fde8760763a8874e182d4f005658',
 'component_bootstrap.py':'4a356c9bb29965afcdb3b704d3959c68b2f18a03401c69dfaddd387eebb38fe9'}
PRIOR_READY_SHA='62347bb0d64d5d8fb816b8c233cdda41be30f89bb92d0a86bc30264f5f3ee639'
WEIGHTS_SHA='f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4'

def require(value,message):
    if not value:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def safe_file(path):
    path=Path(path).absolute()
    require(path.is_file() and not path.is_symlink() and all(not p.is_symlink() for p in path.parents),
        'Missing or symlink file: '+str(path))
    return path

def verifier():
    path=Path(__file__).with_name('verify_v2_transfer.py')
    require(sha(safe_file(path))==VERIFIER_SHA,'Pinned stdlib archive verifier differs')
    spec=importlib.util.spec_from_file_location('_pinned_v3_replay_zip_verifier',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def verify_tree(root,m):
    root=Path(root).absolute();require(root.is_dir() and not root.is_symlink(),'Extracted directory required')
    manifest=read(safe_file(root/'MANIFEST.json'))
    actual=set()
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Symlink inside extracted payload')
        if path.is_file():actual.add(path.relative_to(root).as_posix())
    require(actual==set(manifest['files'])|{'MANIFEST.json'},'Unexpected extracted payload membership')
    for name,item in manifest['files'].items():
        m.safe_member(name);path=safe_file(root/name)
        require(path.resolve().is_relative_to(root.resolve()) and path.stat().st_size==item['bytes']
            and sha(path)==item['sha256'],'Extracted payload changed: '+name)
    return manifest

def context(args,extract=False):
    m=verifier();record=read(safe_file(args.export_record));core_spec=record['archives']['research_core.zip']
    ref_spec=record['archives']['replay_references.zip']
    for path,spec in [(args.core_zip,core_spec),(args.references_zip,ref_spec)]:
        path=safe_file(path)
        require(path.stat().st_size==spec['bytes'] and sha(path)==spec['sha256'],'Delivered ZIP byte identity differs')
    core=args.output/'inputs/core';refs=args.output/'inputs/references'
    for archive,destination,spec in [(args.core_zip,core,core_spec),(args.references_zip,refs,ref_spec)]:
        manifest,internal_sha=m.verify_zip(archive,spec['manifest_sha256'])
        require(manifest['metadata']['study_id']=='generative_recovery_1500','Wrong archive study')
        if extract:m.extract_verified(archive,destination,manifest)
        require(sha(safe_file(destination/'MANIFEST.json'))==internal_sha,'Extracted internal manifest differs')
        verify_tree(destination,m)
    ref=read(refs/'REFERENCE_RECORD.json')
    require(ref['status']=='EXACT_SELECTED_REAL_REFERENCE_AND_SHARED_LATENT_BANKS'
        and ref['core_archive']['sha256']==core_spec['sha256']
        and ref['core_archive']['manifest_sha256']==core_spec['manifest_sha256'],'Reference/core binding differs')
    expected_refs={phase+'_evaluation/'+name for phase in ['validation','test']
        for name in ['real_features.npz','real_features.npz.json','shared_latents.npy']}
    require(set(ref['files'])==expected_refs,'Only exact selected real/z banks accepted')
    for relative,item in ref['files'].items():
        m.safe_member(relative);path=safe_file(refs/relative)
        require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Delivered reference bank differs')
    require(set(ref['producer_commit_record'])=={'validation','test'},'Both producer commit records required')
    for phase,item in ref['producer_commit_record'].items():
        require(item['path'] in {phase+'_evaluation/artifact_registry.json',phase+'_evaluation/committed_artifacts.jsonl'}
            and sha(safe_file(core/item['path']))==item['sha256'],'Reference producer commit binding differs')
    for name,digest in FROZEN.items():require(sha(core/'src'/name)==digest,'Replay metric/model source changed: '+name)
    plan=read(core/'study_plan_frozen.json');selection=read(core/'selection_frozen.json')
    summary=read(core/'confirmation_summary.json');frozen=read(core/'confirmation_plan_frozen.json')
    require(plan['study_id']=='generative_recovery_1500' and sha(core/'study_plan_frozen.json')==ref['study_plan_sha256']
        and sha(core/'selection_frozen.json')==ref['selection_sha256']
        and sha(core/'data/data_ready.json')==ref['data_ready_sha256'],'Actual study/selection/data binding differs')
    ready=read(core/'data/data_ready.json')
    require(ready['study_id']=='generative_recovery_2500' and ready['prior_definition_ready_sha256']==PRIOR_READY_SHA
        and ready['reconstructed_inputs_only'] is True and ready['test_quality_accessed'] is False
        and ready['clean_index_sha256']=='d278e0a67060e70416314a05cb8ea29d6ccd41b80e22ab90c89226bd847156e1'
        and ready['component_ids_sha256']=='bb039432c9bc102fb7d0555a8d381293671d46c46a62bf624dcc0aab31239e8d',
        'New exact reconstructed input definition differs')
    require(set(selection['selected'])=={'B','C'} and plan['development_steps']==1500
        and plan['fixed_main_steps_before_any_validation']==1500,'V3 fixed1500 B/C protocol required')
    methods=['B','C']+(['C0_matched'] if selection['selected']['C']['kappa']==1. else [])
    expected={name+'_seed'+str(seed) for name in methods for seed in [101,202,303]}
    jobs={job['run_id']:job for job in frozen['jobs']}
    require(len(jobs)==len(frozen['jobs'])==len(expected) and set(jobs)==expected
        and summary['status']=='complete' and summary['planned_runs']==summary['complete_runs']==len(expected)
        and {item['run_id'] for item in summary['runs']}==expected,'All selected fresh3seed main models required')
    index_path=core/'test_evaluation/results_index.json'
    if index_path.exists():
        index=read(index_path)
        require(index['status']=='COMPLETED' and index['point_count']==len(expected),'All original main point records required')
        point_paths=index['points'];point_scope='FINAL_PRODUCER_INDEX'
    else:
        journal=core/'test_evaluation/committed_artifacts.jsonl'
        entries=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        point_paths={Path(item['path']).stem:item['path'] for item in entries
            if item['path'].startswith('points/') and item['path'].endswith('.json')}
        for item in entries:
            if item['path'] in point_paths.values():
                require(sha(core/'test_evaluation'/m.safe_member(item['path']))==item['sha256'],
                    'Incomplete-producer journal point changed')
        point_scope='JOURNAL_ONLY_PARTIAL_PRODUCER_NO_FINAL_INDEX_INVENTED'
    require(set(point_paths)=={key+'_ema'+str(jobs[key]['selected_ema_decay']) for key in expected},
        'All exact selected original main point records required')
    output_jobs=[]
    for entry in sorted(summary['runs'],key=lambda item:item['run_id']):
        key=entry['run_id'];config=jobs[key];method='C' if key.startswith('C0_matched_') else key.rsplit('_seed',1)[0]
        chosen=selection['selected'][method];decay=chosen['ema_decay']
        complete_path=core/'confirmation'/key/'complete.json';complete=read(complete_path)
        require(sha(complete_path)==entry['complete_sha256'] and complete['config']==config
            and complete['status']=='complete' and complete['completed_steps']==config['steps'] in [1500]
            and config['arm']==method and config['coefficient']==chosen['coefficient']
            and config['kappa']==(0. if key.startswith('C0_matched_') else chosen['kappa'])
            and config['seed']==int(key.rsplit('_seed',1)[1])
            and config['selected_ema_decay']==decay and config['selection_sha256']==sha(core/'selection_frozen.json'),
            'Original completed-model metadata differs')
        folder=args.models_root/key;model_manifest=verify_tree(folder,m)
        required={key+'_EMA.pt','models.py','generate.py','LICENSE_NOTICE.txt','config.json','training_complete.json','MODEL_RECORD.json'}
        require(set(model_manifest['files'])==required,'Unexpected delivered model payload')
        original_record=core/'durable'/key/'payload/MODEL_RECORD.json';model_record=read(folder/'MODEL_RECORD.json')
        require(sha(folder/'MODEL_RECORD.json')==sha(original_record)
            and model_record['study_plan_sha256']==sha(core/'study_plan_frozen.json')
            and model_record['complete_record_sha256']==entry['complete_sha256']
            and model_record['original_full_checkpoint_sha256']==complete['checkpoint_sha256']
            and model_record['selected_ema_decay']==decay,'Delivered model record differs from actual core')
        ema=safe_file(folder/(key+'_EMA.pt'))
        require(sha(ema)==model_record['EMA_sha256']==complete['ema_artifacts'][str(decay)]['sha256']
            and ema.stat().st_size==model_record['EMA_bytes']
            and sha(folder/'models.py')==FROZEN['models.py']
            and read(folder/'config.json')==config and read(folder/'training_complete.json')==complete,
            'Actual delivered EMA/source/config bytes differ')
        point_key=key+'_ema'+str(decay);point_path=core/'test_evaluation'/m.safe_member(point_paths[point_key])
        point=read(point_path)
        require(point['EMA_sha256']==sha(ema) and point['complete_sha256']==entry['complete_sha256']
            and point['config']==config and point['real_feature_sha256']==sha(refs/'test_evaluation/real_features.npz')
            and point['shared_latents_sha256']==sha(refs/'test_evaluation/shared_latents.npy'),
            'Original main point/reference binding differs')
        output_jobs.append({'run_id':key,'arm':config['arm'],'seed':config['seed'],'config':config,
            'ema_decay':decay,'EMA_path':str(ema),'EMA_sha256':sha(ema),'MODEL_RECORD_sha256':sha(folder/'MODEL_RECORD.json'),
            'source_fullCP_sha256_RECORDED_ONLY_NOT_REQUIRED':complete['checkpoint_sha256'],
            'producer_point_path':str(point_path),'producer_point_sha256':sha(point_path)})
    return {'core':str(core),'references':str(refs),'methods':methods,'jobs':output_jobs,
        'core_zip_sha256':core_spec['sha256'],'references_zip_sha256':ref_spec['sha256'],
        'export_record_sha256':sha(args.export_record),'producer_point_commit_scope':point_scope,'study_plan_sha256':sha(core/'study_plan_frozen.json')}


def worker(args):
    began=time.monotonic()
    bound=read(args.output/'PREFLIGHT.json');ctx=context(args)
    require(ctx==bound['context'] and sha(__file__)==bound['replay_source_sha256'],'Replay inputs/source changed')
    import numpy as np
    import torch
    sys.path.insert(0,str(Path(ctx['core'])/'src'))
    from models import Generator
    from clean_features import CleanFIDFeatures
    from evaluate import rgb_uint8,evaluate_feature_groups
    from diversity import evaluate_prdc_groups
    from component_bootstrap import component_paired_bootstrap
    require(sha(safe_file(args.weights_dir/'inception-2015-12-05.pt'))==WEIGHTS_SHA,'Expected external feature weights absent/changed')
    extractor=CleanFIDFeatures(device=args.device,weights_dir=str(args.weights_dir),allow_download=False,
        expected_weights_sha256=WEIGHTS_SHA)
    refs=Path(ctx['references']);z=np.load(refs/'test_evaluation/shared_latents.npy',allow_pickle=False)
    require(z.shape==(512,128) and z.dtype==np.float32 and np.isfinite(z).all(),'Wrong actual shared latent bank')
    with np.load(refs/'test_evaluation/real_features.npz',allow_pickle=False) as bank:
        real={key:bank[key].copy() for key in bank.files}
    real['provenance']=read(refs/'test_evaluation/real_features.npz.json')['provenance']
    require(real['provenance']['feature_extractor']==extractor.provenance,
        'Replay extractor/package environment differs from saved benchmark provenance')
    jobs=[next(job for job in ctx['jobs'] if job['run_id']=='C_seed101')] if args.smoke_only else ctx['jobs']
    fakes={};points={}
    for job in jobs:
        require(sha(safe_file(job['EMA_path']))==job['EMA_sha256'],'EMA changed immediately before safe load')
        payload=torch.load(job['EMA_path'],map_location='cpu',weights_only=True)
        cfg=payload['generator_config']
        require(payload['study_id']=='generative_recovery_1500' and payload['config']==job['config']
            and payload['ema_decay']==job['ema_decay'] and payload['step']==job['config']['steps']
            and payload['arm']==job['arm'] and payload['seed']==job['seed']
            and payload['source_sha256']==job['config']['source_sha256']
            and cfg['z_dim']==128 and cfg['num_classes']==4 and cfg['base_channels']==32 and cfg['image_size']==64,
            'Safe EMA payload/common architecture differs')
        g=Generator(z_dim=128,num_classes=4,base_channels=32).to(args.device)
        g.load_state_dict(payload['ema_generator'],strict=True);g.eval().requires_grad_(False)
        batches=[];groups=[];ids=[];latent=[]
        for group in ([0] if args.smoke_only else range(4)):
            for start in range(0,64 if args.smoke_only else 512,64):
                with torch.inference_mode():
                    images=g(torch.from_numpy(z[start:start+64]).to(args.device),
                        torch.full((64,),group,dtype=torch.long,device=args.device))
                pixels=rgb_uint8(images.float().cpu().numpy().transpose(0,2,3,1));batches.append(extractor(pixels))
                groups.extend([group]*64);latent.extend(range(start,start+64))
                ids.extend([f'test:z94001:g{group}:i{i:06d}' for i in range(start,start+64)])
        features=np.concatenate(batches)
        if args.smoke_only:
            require(features.shape==(64,2048) and np.isfinite(features).all(),'Onebatch fake feature smoke failed')
            np.save(args.output/'C_seed101_group0_first64_features.npy',features.astype(np.float32),allow_pickle=False)
            require(context(args)==ctx,'Replay inputs changed during onebatch smoke')
            write(args.output/'SMOKE.json',{'status':'SMOKE_PASSED_NO_KID_OR_PRDC_OR_CI_COMPUTED',
                'run_id':job['run_id'],'EMA_sha256':job['EMA_sha256'],'feature_shape':[64,2048],
                'features_finite':True,'strict_G_load':True,'weights_only':True,'source':'REPLAY_NOT_ORIGINAL_PRODUCER'})
            return
        fake={'features':features,'groups':np.asarray(groups,dtype=np.int64),'ids':np.asarray(ids),
            'latent_indices':np.asarray(latent,dtype=np.int64),'provenance':{'feature_extractor':extractor.provenance}}
        path=args.output/'features'/(job['run_id']+'.npz');path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,**{key:(value.astype(np.float32) if key=='features' else value)
            for key,value in fake.items() if key!='provenance'})
        kid=evaluate_feature_groups(real,fake,analysis_cap=512,selection_seed=95001)
        prdc=evaluate_prdc_groups(real,fake,nearest_k=3,analysis_cap=512,selection_seed=95001,kid_point_report=kid)
        original=read(job['producer_point_path'])
        point={'status':'REPLAY_POINT_NOT_ORIGINAL_PRODUCER','run_id':job['run_id'],'EMA_sha256':job['EMA_sha256'],
            'producer_point_sha256':job['producer_point_sha256'],'kid':kid,'prdc':prdc,
            'feature_npz_sha256':sha(path),'raw_KID_difference_from_producer':
                [kid['groups'][str(group)]['kid']-original['kid']['groups'][str(group)]['kid'] for group in range(4)],
            'bitwise_or_numerical_agreement_not_assumed':True}
        write(args.output/'points'/(job['run_id']+'.json'),point);points[job['run_id']]=point;fakes[job['run_id']]=fake
        del g,payload
        if torch.cuda.is_available():torch.cuda.empty_cache()
    intervals={}
    if args.with_ci:
        for reference in ['B']+(['C0_matched'] if 'C0_matched' in ctx['methods'] else []):
            report=component_paired_bootstrap(real,[fakes['C_seed'+str(s)] for s in [101,202,303]],
                [fakes[reference+'_seed'+str(s)] for s in [101,202,303]],replicates=2000,seed=95003,
                confidence=.9875,batch_size=32,max_seconds=min(600,max(1,args.max_seconds-(time.monotonic()-began)-10)))
            write(args.output/'bootstrap'/('C_vs_'+reference+'.json'),report);intervals[reference]=report['status']
    require(context(args)==ctx,'Replay inputs changed during feature/point/CI computation')
    write(args.output/'REPLAY_RESULTS.json',{'status':'REPLAY_POINTS_COMPUTED_NOT_ORIGINAL_PRODUCER_RECORDS',
        'point_count':len(points),'planned_points':len(ctx['jobs']),'CI_requested':args.with_ci,
        'CI_status':intervals if args.with_ci else 'NOT_RUN_EXPLICIT_FLAG_OMITTED',
        'all_requested_CIs_computed':(all(v=='computed' for v in intervals.values()) if args.with_ci else None),
        'fixed_seed_population_inference':False,'real_reference_bank_reextracted':False,
        'full_DEV10_regeneration_from_selected_main_EMAs':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['models-root','core-zip','references-zip','export-record','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--weights-dir',type=Path,default=Path('/tmp'));parser.add_argument('--device',default='cuda:0')
    group=parser.add_mutually_exclusive_group();group.add_argument('--preflight-only',action='store_true');group.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--with-ci',action='store_true');parser.add_argument('--max-seconds',type=int,default=1200)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    require(sys.flags.optimize==0 and 30<=args.max_seconds<=3600,'Nonoptimized finite replay budget required')
    require(not (args.with_ci and (args.preflight_only or args.smoke_only)),'CI only in full replay mode')
    for name in ['output','models_root','core_zip','references_zip','export_record','weights_dir']:
        setattr(args,name,getattr(args,name).absolute())
    if args.worker:return worker(args)
    require(not args.output.exists() and not args.output.is_symlink()
        and all(not p.is_symlink() for p in args.output.parents),'Fresh output required')
    args.output.mkdir(parents=True);began=time.monotonic();process=None
    def stop():
        if process is not None and process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait(timeout=5)
    def timeout(*_):raise TimeoutError('Finite portable replay cap; no original producer overwritten')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(args.max_seconds)
    try:
        ctx=context(args,extract=True)
        write(args.output/'PREFLIGHT.json',{'status':'PREFLIGHT_PASSED_NO_TORCH_OR_MODEL_LOAD',
            'context':ctx,'replay_source_sha256':sha(__file__),'historical_study_clock_not_reset_or_reapplied':True})
        if args.preflight_only:return
        worker_seconds=int(args.max_seconds-(time.monotonic()-began))-5
        require(worker_seconds>=30,'No finite worker budget remains after preflight')
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
            '--models-root',str(args.models_root),'--core-zip',str(args.core_zip),
            '--references-zip',str(args.references_zip),'--export-record',str(args.export_record),
            '--output',str(args.output),'--weights-dir',str(args.weights_dir),'--device',args.device,
            '--max-seconds',str(worker_seconds)]
        if args.smoke_only:command.append('--smoke-only')
        if args.with_ci:command.append('--with-ci')
        with (args.output/'worker.log').open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=Path(__file__).parent)
            code=process.wait(timeout=max(1,args.max_seconds-(time.monotonic()-began)))
        require(code==0,'Replay worker failed; preserve new partial tree')
        if args.smoke_only:status='REPLAY_SMOKE_COMPLETE_NO_QUALITY'
        else:
            result=read(args.output/'REPLAY_RESULTS.json')
            status='REPLAY_POINTS_COMPLETE_CI_INCOMPLETE' if args.with_ci and not result['all_requested_CIs_computed'] else 'REPLAY_COMPLETE'
        write(args.output/'COMPLETE.json',{'status':status,
            'elapsed_seconds':time.monotonic()-began,'source_sha256':sha(__file__),'original_producer_records_overwritten':False})
    except BaseException as error:
        signal.alarm(0);stop()
        write(args.output/'FAILURE.json',{'status':'REPLAY_INCOMPLETE_PRESERVED','error_type':type(error).__name__})
        raise
    finally:signal.alarm(0);stop()

if __name__=='__main__':main()
