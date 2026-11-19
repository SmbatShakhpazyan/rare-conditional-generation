"""CPU-only compact V3 core, exact real references, auxiliary teacher and derived split ZIPs.

No Torch/NumPy/model/quality computation, network, raw face or raw identity.
Full derived component indices are explicitly confined to separate private split support. Only explicit owned source-bound artifact paths.
"""
from __future__ import annotations
import argparse
import ast
import base64
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path,PurePosixPath
import re
import signal
import struct
import sys
import tempfile
import zipfile

ROOT=Path('/kaggle/working/generative_recovery_2500')
GENERIC_SHA='34ebd3878e48c8afdccd1080453a29a25c536ac12495c4e5ed1621a58bf3bd4c'
CAPS={524288,1048576,2097152}
NOTICE='''PRIVATE RESEARCH REPRODUCIBILITY ARTIFACT; NO BLANKET LICENSE GRANT

Fresh fixed1500 component-split V3 is separate from historical V1 and lost V2 cohorts. Outcomes,
failures and interval status are recorded in RESEARCH_STATUS.json; a cloud
archive being built is not proof its bytes have reached the recipient.

No real face pixels, original JPEG/ZIP/CSV/TXT metadata, supplied identity labels,
full-N private component map, near-edge/descriptor arrays, authentication state,
pretrained benchmark weights, optimizer or full GAN training checkpoint is
included. Real feature banks contain ONLY selected held-out reference features,
public dataset filenames/row indices and selected-bank union-component labels
needed to replay joint cluster uncertainty. These are the exact producer NPZ
bytes, not a complete identity database or a biometric identity inference.
Generated PNG grids are fixed synthetic samples, never selected by appearance.
The separate teacher ZIP contains a NEW auxiliary discriminator-only snapshot
trained on clean TRAIN; it is not an externally pretrained face/identity model.
Inference generators travel in separate per-model selected-EMA archives.

CelebA https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html restricts use to
non-commercial research with redistribution restrictions. This private delivery
grants no new dataset, identity, publicity, trained-weight or commercial rights.
Dependencies retain their licenses. Weights/images are not certified free of
memorization. Literal condition annotations and image-feature metrics imply no
semantic, fairness, privacy, universal quality, priority or convergence guarantee.
No attribute judge is used in this V3 study. No lost V2 model or metric is claimed recovered. Known feature matching/MMD,
normalization/tolerance/path-norm/gSR antecedents remain credited separately.
'''


def require(value,message):
    if not value:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def owned(root,value):
    value=Path(value)
    if not value.is_absolute():value=root/value
    require('..' not in value.parts and not value.is_symlink(),'Unsafe/symlink artifact')
    require(all(not p.is_symlink() for p in value.absolute().parents),'Symlink parent refused')
    value=value.resolve();require(value.is_relative_to(root),'Artifact escaped owned root')
    return value

def regular(path,maximum=None):
    require(path.is_file() and not path.is_symlink(),'Missing/alias artifact')
    require(maximum is None or path.stat().st_size<=maximum,'Artifact exceeds declared size cap')
    return path

def generic(root):
    path=root/'runtime/export_results.py';require(sha(path)==GENERIC_SHA,'Frozen generic ZIP helper changed')
    spec=importlib.util.spec_from_file_location('_verified_v3_generic_zip',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def npy_header(stream):
    require(stream.read(6)==b'\x93NUMPY','Not NPY format')
    version=stream.read(2)
    require(version in (b'\x01\x00',b'\x02\x00',b'\x03\x00'),'Unsupported NPY version')
    width=2 if version==b'\x01\x00' else 4
    length=struct.unpack('<H' if width==2 else '<I',stream.read(width))[0]
    require(length<=16384,'Oversized NPY header')
    result=ast.literal_eval(stream.read(length).decode('utf8' if version==b'\x03\x00' else 'latin1'))
    require(isinstance(result,dict) and set(result)=={'descr','fortran_order','shape'}
        and result['fortran_order'] is False and isinstance(result['shape'],tuple)
        and isinstance(result['descr'],str) and 'O' not in result['descr'],'Unsafe NPY header')
    return result

def feature_bank_check(path,real):
    regular(path,48*1024**2)
    expected={'features','groups','ids','component_ids','original_rows'} if real else {'features','groups','ids','latent_indices'}
    with zipfile.ZipFile(path) as z:
        require(set(z.namelist())=={n+'.npy' for n in expected} and len(z.namelist())==len(expected),
            'Feature NPZ contains unexpected/private arrays')
        headers={}
        for name in expected:
            with z.open(name+'.npy') as stream:
                h=npy_header(stream);headers[name]=h;header_bytes=stream.tell()
            require(all(type(d) is int and d>=0 for d in h['shape']),'Invalid array dimensions')
            kind=re.fullmatch(r'[<=>|]([fiubSU])([1-9][0-9]*)',h['descr'])
            require(kind is not None,'Unrecognized numeric/string dtype')
            itemsize=int(kind[2])*(4 if kind[1]=='U' else 1)
            require(z.getinfo(name+'.npy').file_size==header_bytes+math.prod(h['shape'])*itemsize,
                'Truncated or excess NPY array body')
        n=headers['features']['shape'][0]
        require(headers['features']['shape']==(n,2048) and headers['features']['descr'] in ('<f4','=f4')
            and 400<=n<=2048,'Feature bank is not bounded selected float32 Inception rows')
        if not real:require(n==2048,'Exactly512x4 fake features required')
        require(all(h['shape']==(n,) for name,h in headers.items() if name!='features'),'Feature metadata rows differ')
        require(headers['ids']['descr'].startswith(('<U','|S')),'String sample IDs required')
        for name in expected-{'features','ids'}:
            require(headers[name]['descr'] in ('<i8','<i4','|i1','<i2','<u8','<u4','|u1','<u2'),
                'Feature metadata must be integer vectors')
    side=read(Path(str(path)+'.json'))
    require(side['npz_sha256']==sha(path) and side['array_rows']==n and side['feature_dimension']==2048
        and side['feature_dtype']=='float32_lossless_from_extractor_float32','Feature sidecar differs')
    return n

def exposure_check(path,steps):
    with path.open('rb') as f:
        header=npy_header(f)
        require(header['shape']==(202599,) and header['descr']=='<i8','Only presentation-count exposure vector allowed')
        data=f.read()
    require(len(data)==202599*8,'Wrong exposure byte size')
    total=0
    for (value,) in struct.iter_unpack('<q',data):
        require(value>=0,'Negative exposure');total+=value
    require(total==steps*64,'Actual exposure sum differs')

def armour(archive,folder,limit):
    require(limit in CAPS,'Unknown transport cap')
    record={'archive_name':archive.name,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
        'max_encoded_bytes_per_chunk':limit,'columns':76,'encoding':'base64_ASCII_LF_final_newline','chunks':[]}
    prefix=archive.stem+'--'+record['archive_sha256'][:16]
    target=folder/(prefix+'.b64chunks');target.mkdir(exist_ok=False)
    raw_size=(limit//77)*57;whole=hashlib.sha256();count=0
    with archive.open('rb') as f:
        for number,raw in enumerate(iter(lambda:f.read(raw_size),b''),1):
            encoded=base64.b64encode(raw);text=b'\n'.join(encoded[i:i+76] for i in range(0,len(encoded),76))+b'\n'
            require(len(text)<=limit and text.isascii(),'Encoded chunk cap/ASCII mismatch')
            name=f'{prefix}--chunk{number:05d}.b64.txt';path=target/name
            with path.open('xb') as out:out.write(text)
            decoded=base64.b64decode(path.read_bytes().replace(b'\n',b''),validate=True)
            require(decoded==raw,'Perchunk roundtrip differs');whole.update(decoded);count+=len(decoded)
            record['chunks'].append({'filename':name,'bytes':len(text),'sha256':sha(path),
                'decoded_bytes':len(raw),'decoded_sha256':hashlib.sha256(raw).hexdigest()})
    require(whole.hexdigest()==record['archive_sha256']==sha(archive) and count==record['archive_bytes'],
        'Ordered whole ASCII reconstruction differs')
    record['chunk_count']=len(record['chunks']);path=target/'CHUNKS.json';write(path,record)
    return path


def complete_report(expected,main,val,test,refs):
    boot=test.get('bootstrap',{})
    return (bool(expected) and main.get('status')=='complete'
        and main.get('planned_runs')==main.get('complete_runs')==len(expected)
        and {r['run_id'] for r in main.get('runs',[])}==expected
        and val.get('status')=='COMPLETED' and val.get('point_count')==10
        and test.get('status')=='COMPLETED' and test.get('point_count')==len(expected)
        and set(boot)==refs and all(value['status']=='computed' for value in boot.values())
        and test.get('complete_scientific_report') is True)


def collect(root):
    plan=read(root/'study_plan_frozen.json')
    require(plan['study_id']=='generative_recovery_1500' and plan['status']=='frozen_before_teacher_grid_validation_test',
        'Wrong v3 study plan')
    ready=read(root/'data/data_ready.json')
    require(sha(root/'data/data_ready.json')==plan['data_ready_sha256'] and ready['status']=='frozen_clean_component_split'
        and ready['study_id']=='generative_recovery_2500' and ready['reconstructed_inputs_only'] is True
        and ready['prior_definition_ready_sha256']=='62347bb0d64d5d8fb816b8c233cdda41be30f89bb92d0a86bc30264f5f3ee639'
        and ready['test_quality_accessed'] is False and ready['original_val_test_images_used'] is False,
        'Clean split provenance changed')
    files={};features={};teacher_files={};frozen={}
    def add(name,required=True,target=None):
        path=owned(root,name)
        if not path.exists():
            require(not required,'Required research artifact missing: '+str(name));return
        regular(path);label=Path(name).as_posix() if not Path(name).is_absolute() else path.relative_to(root).as_posix()
        (files if target is None else target)[label]=path;frozen[str(path)]=sha(path)
    for name,digest in plan['training_sources'].items():
        require(Path(name).name==name and sha(root/'src'/name)==digest,'Training source changed');add('src/'+name)
    for name,digest in plan['auxiliary_sources'].items():
        require(sha(owned(root,name))==digest,'Auxiliary source changed');add(name)
    approval=read(root/'evaluation_sources_approved.json')
    for name,digest in approval['sources'].items():
        require(Path(name).name==name and sha(root/'src'/name)==digest,'Evaluation source changed');add('src/'+name)
    for name in ['runtime/export_v3_compact.py','runtime/replay_main_ema_v3.py','runtime/export_results.py','runtime/verify_v2_transfer.py']:
        add(name)
    for name in ['study_plan_frozen.json','study_clock.json','budget_frozen_before_validation.json',
        'development_summary.json','selection_frozen.json','confirmation_plan_frozen.json','confirmation_summary.json',
        'training_sources_approved.json','confirmation_sources_approved.json','evaluation_sources_approved.json',
        'export_sources_approved.json','data/data_ready.json','data/split_policy.json','data/split_report.json',
        'data/near_screen_summary.json']:
        add(name,required=name in {'study_plan_frozen.json','study_clock.json','budget_frozen_before_validation.json',
            'training_sources_approved.json','evaluation_sources_approved.json','export_sources_approved.json','data/data_ready.json'})
    for name,field in [('data/split_policy.json','split_policy_sha256'),('data/split_report.json','split_report_sha256'),
                       ('data/near_screen_summary.json','near_screen_summary_sha256')]:
        require(sha(root/name)==ready[field],'Data aggregate changed');add(name)
    require(read(root/'budget_frozen_before_validation.json')['main_steps']==1500
        and plan['development_steps']==1500 and plan['fixed_main_steps_before_any_validation']==1500,
        'Only separate fixed1500 V3 protocol accepted')
    for name in ['data_preparation_approved.json','preparation_guard/started.json','preparation_guard/process.json',
        'preparation_guard/stdout.log','preparation_guard/stderr.log','integrity/canonical_metadata.json',
        'integrity/identity_acquisition.json','integrity/exact_flip_summary.json']:
        add(name,required=False)
    for name,digest in ready.get('pure_sources',{}).items():
        require(sha(owned(root,name))==digest,'Preparation pure source changed');add(name)
    prep=ready['preparation_source_path'];require(sha(owned(root,prep))==ready['source_sha256'],'Actual split source changed');add(prep)
    # The actual input recovery and all-phase controller are distinct from vanished V2 state.
    if ready.get('recovery_method')=='one_authenticated_public_ZIP_local_bytes_no_FUSE_JPEG_reads':
        require(sha(root/'integrity/archive_download.json')==ready['archive_download_sha256']
            and sha(root/'integrity/archive_selection.json')==ready['archive_selection_sha256']
            and sha(root/'preparation_guard/process.json')==ready['prior_failed_process_sha256']
            and sha(root/'runtime/prepare_recovery_data.py')==ready['prior_failed_FUSE_source_sha256'],
            'Actual new ZIP recovery or preserved failed-attempt proof changed')
        zip_approval=read(root/'zip_data_preparation_approved.json')
        require(sha(root/'zip_data_preparation_approved.json')==ready['approval_sha256']
            and zip_approval['source_sha256']==ready['source_sha256']
            and sha(root/'runtime/fuse_preparation_failure_observed.json')==zip_approval['observed_failure_sha256'],
            'Actual ZIP approval/observed failure binding differs')
        terminal=read(root/'zip_preparation_guard/process.json')
        require(terminal['status']=='COMPLETED' and terminal['return_code']==0
            and terminal['source_sha256']==ready['source_sha256']
            and terminal['ready_sha256']==sha(root/'data/data_ready.json'),'Accepted terminal input recovery required')
        for name in ['runtime/prepare_recovery_data.py','runtime/fuse_preparation_failure_observed.json',
            'zip_data_preparation_approved.json','data_preparation_approved.json','integrity/archive_download.json',
            'integrity/archive_selection.json','zip_preparation_guard/started.json','zip_preparation_guard/process.json',
            'zip_preparation_guard/preserved_failed_attempt.json','zip_preparation_guard/stdout.log',
            'zip_preparation_guard/stderr.log']:
            add(name,required=False)
    if (root/'supervisor_sources_approved.json').exists():
        supervisor_approval=read(root/'supervisor_sources_approved.json')
        require(supervisor_approval['study_id']=='generative_recovery_1500'
            and supervisor_approval['main_steps']==1500
            and sha(root/'runtime/supervise_v3_study.py')==supervisor_approval['supervisor_sha256'],
            'Actual all-phase source approval changed')
        add('supervisor_sources_approved.json');add('runtime/supervise_v3_study.py')
        for name in ['started.json','actual_ready_bound.json','development.started.json','development.process.json',
            'development.log','validation.started.json','validation.process.json','validation.log',
            'confirmation.started.json','confirmation.process.json','confirmation.log','test.started.json',
            'test.process.json','test.log','complete.json','failure.json']:
            add('runtime/supervisor/'+name,required=False)
    for name,digest in {
        'prior_audit/v2_data_ready.json':'62347bb0d64d5d8fb816b8c233cdda41be30f89bb92d0a86bc30264f5f3ee639',
        'prior_audit/attempt002_summary.json':'d58e5e4614db05522ff0ee160c18e585573ccc3d92fc95ca7c065fc93a177689',
        'prior_audit/actual_exposure_summary.json':'85a22d45c56098ef604a0a5b0e285c859a23734719216857cec030ac0c0f4a5a'}.items():
        require(sha(owned(root,name))==digest,'Recorded historical input/audit aggregate changed');add(name)
    # Explicit known aggregate/source paths only; never recurse through audit/private.
    for directory in ['audit/attempt001','audit/attempt002','audit/attempt001_guard','audit/attempt002_guard',
        'audit/near_attempt001_guard','identity/attempt001','audit/exposure_attempt001','runtime/exposure_attempt001']:
        for name in ['summary.json','started.json','failure.json','process.json','identity_aggregate.json',
            'filenames_and_exposures.json','exact_pixel_aggregate.json','acquisition.json','acquisition_failure.json',
            'identity_acquisition.json','identity_child.process.json','worker_attempt_started.json','stdout.log','stderr.log']:
            add(directory+'/'+name,required=False)
    for name in ['audit/audit_exact_identity.py','audit/audit_exact_identity_recovery.py','audit/run_exact_identity_bounded.py',
        'audit/run_exact_identity_recovery_bounded.py','audit/build_clean_component_split.py','audit/run_clean_component_split_bounded.py',
        'audit/acquire_identity_mirror.py','audit/acquire_identity_mirror_actual_attempt001.py',
        'audit/mirror_source_chronology.json','audit/audit_actual_exposure_components.py','audit/run_actual_exposure_bounded.py',
        'data_audit_sources_approved.json','data_audit_recovery_sources_approved.json','identity_mirror_source_approved.json',
        'near_split_sources_approved.json','actual_exposure_sources_approved.json']:
        add(name,required=False)
    for path in sorted((root/'logs').glob('*')):
        if path.suffix in {'.log','.json'}:regular(path,16*1024**2);add(path)
    for phase in ['development','pilot','confirmation']:
        for folder in sorted((root/phase).glob('*')):
            if not folder.is_dir():continue
            require(not folder.is_symlink(),'Run directory alias refused')
            for name in ['start.json','complete.json','failure.json','training.jsonl']:add(folder/name,required=False)
            if (folder/'complete.json').exists() and phase!='pilot':
                result=read(folder/'complete.json');config=result['config']
                require(result['status']=='complete' and config['source_sha256']==plan['training_sources']
                    and config['data_ready_sha256']==plan['data_ready_sha256'],'Completed run/source/data differs')
                require(sha(folder/'latest.pt')==result['checkpoint_sha256'],'Source fullCP changed')
                exposure_check(folder/'exposures.npy',config['steps']);add(folder/'exposures.npy')
                add('configs/'+phase+'/'+folder.name+'.json')
                require(read(root/'configs'/phase/(folder.name+'.json'))==config,'Saved config differs from actual run')
                if phase=='confirmation':
                    durable=root/'durable'/folder.name
                    add(durable/'complete.json');add(durable/'payload/MODEL_RECORD.json')
                    record=read(durable/'complete.json')
                    require(record['status']=='durable_selected_EMA_archive_and_text_chunks_verified'
                        and record['run']==folder.name and record['source_sha256']==sha(root/'runtime/archive_v3_model.py'),
                        'Immediate durable model record differs')
                    ema=folder/('EMA_'+str(config['selected_ema_decay'])+'.pt')
                    require(sha(ema)==record['EMA_sha256']==result['ema_artifacts'][str(config['selected_ema_decay'])]['sha256'],
                        'Selected delivered EMA binding changed')
    for phase,summary_name in [('development','development_summary.json'),('confirmation','confirmation_summary.json')]:
        if not (root/summary_name).exists():continue
        summary=read(root/summary_name);entries=summary['runs']
        require(len(entries)==len({entry['run_id'] for entry in entries}),'Duplicate summary runs')
        for entry in entries:
            folder=root/phase/entry['run_id'];complete_path=folder/'complete.json'
            require(sha(complete_path)==entry['complete_sha256'],'Summary completion hash changed')
            if phase=='confirmation':
                require(sha(root/'durable'/entry['run_id']/'complete.json')==entry['durable_complete_sha256'],
                    'Summary durable completion changed')
                jobs=read(root/'confirmation_plan_frozen.json')['jobs']
                matching=[job for job in jobs if job['run_id']==entry['run_id']]
                require(len(matching)==1 and read(complete_path)['config']==matching[0],
                    'Actual confirmation config differs from frozen job')
    phase_records={};phase_points={}
    for phase in ['validation','test']:
        base=root/(phase+'_evaluation');registry_path=base/'artifact_registry.json'
        journal=base/'committed_artifacts.jsonl'
        if not registry_path.exists() and not journal.exists():
            phase_records[phase]={'status':'MISSING_OR_UNCOMMITTED','point_count':0,'bootstrap':{}}
            phase_points[phase]=[]
            for name in ['failure.json','binding_before_quality.json']:add(base/name,required=False)
            continue
        require((root/(phase+'_evaluation_guard')/'process.json').exists(),'Evaluation must be terminal before export')
        final_registry=registry_path.exists()
        if final_registry:
            registry=read(registry_path);require(registry['status']=='committed','Uncommitted phase registry')
            add(registry_path)
        else:
            entries=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
            require(len(entries)==len({item['path'] for item in entries}),'Duplicate journal paths')
            registry={'status':'journal_only_partial','sha256':{item['path']:item['sha256'] for item in entries}}
        phase_points[phase]=[]
        for relative,digest in registry['sha256'].items():
            path=owned(base,relative);require(sha(path)==digest,'Committed evaluation artifact changed')
            parts=PurePosixPath(relative).parts
            if parts[0]=='points' and path.suffix=='.json':
                point=read(path);phase_folder='development' if phase=='validation' else 'confirmation'
                folder=root/phase_folder/point['run_id'];complete=read(folder/'complete.json')
                require(point['config']==complete['config'] and point['phase']==phase
                    and point['complete_sha256']==sha(folder/'complete.json')
                    and point['step']==complete['completed_steps']
                    and point['real_feature_sha256']==registry['sha256']['real_features.npz']
                    and point['shared_latents_sha256']==registry['sha256']['shared_latents.npy']
                    and point['feature_definition']['feature_dim']==2048
                    and point['feature_definition']['weights_sha256']=='f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4'
                    and point['study_plan_sha256']==sha(root/'study_plan_frozen.json'),
                    'Point/source/reference binding differs')
                item=complete['ema_artifacts'][str(point['ema_decay'])]
                require(point['EMA_path']==item['path'] and point['EMA_sha256']==item['sha256']
                    and sha(owned(root,item['path']))==item['sha256'],'Point selectedEMA binding differs')
                fake_name='features/fake_'+point['run_id']+'_ema'+str(point['ema_decay'])+'.npz'
                require(point['fake_feature_sha256']==registry['sha256'][fake_name],
                    'Point fakefeature binding differs')
                phase_points[phase].append(point)
            if relative.endswith('.npz'):
                require(relative=='real_features.npz' or (parts[0]=='features' and len(parts)==2
                    and re.fullmatch(r'fake_[A-Za-z0-9_]+_ema0\.(99|999)\.npz',parts[-1])),
                    'Unrecognized NPZ refused')
                feature_bank_check(path,real=relative=='real_features.npz');add(path,target=features)
            elif relative=='shared_latents.npy':
                with path.open('rb') as stream:h=npy_header(stream)
                require(h['shape']==(512,128) and h['descr']=='<f4','Wrong shared synthetic latents');add(path,target=features)
            elif relative.endswith('.npz.json'):
                require(path.with_suffix('').exists(),'Orphan feature sidecar');add(path,target=features)
            elif parts[0]=='generated_grids' and len(parts)==3 and parts[-1] in {'group0.png','group1.png','group2.png','group3.png'}:
                raw=path.read_bytes();require(raw[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',raw[16:24])==(256,256),
                    'Only producerfixed synthetic256px grids allowed');add(path)
            else:
                require(path.suffix=='.json' and (parts[0] in {'points','bootstrap'} or relative in
                    {'binding_before_quality.json','results_index.json'}),'Unrecognized phase artifact refused');add(path)
        for name in ['committed_artifacts.jsonl','failure.json']:add(base/name,required=False)
        for name in ['process.json','stdout.log','stderr.log']:add(root/(phase+'_evaluation_guard')/name,required=False)
        phase_records[phase]=(read(base/'results_index.json') if final_registry else
            {'status':'JOURNAL_ONLY_PARTIAL','point_count':len(phase_points[phase]),'bootstrap':{}})
    teacher=read(root/'teacher/complete.json')
    require(teacher['status']=='complete' and teacher['source_plan_sha256']==sha(root/'study_plan_frozen.json')
        and sha(root/'teacher/fixed_teacher.pt')==teacher['teacher_artifact_sha256']
        and sha(root/'teacher/statistics.json')==teacher['statistics_sha256'],'Actual fixedteacher snapshot changed')
    stats=read(root/'teacher/statistics.json');bcomplete=root/'development/B_seed909/complete.json'
    require(stats['teacher_complete_sha256']==sha(bcomplete)
        and stats['teacher_source_checkpoint_sha256']==read(bcomplete)['checkpoint_sha256']
        and stats['teacher_D_export_sha256']==read(bcomplete)['D_final_sha256'],
        'Fixedteacher provenance differs from exactlast Bdevelopment source')
    for name in ['teacher/statistics.json','teacher/complete.json']:add(name)
    for name in ['teacher/fixed_teacher.pt','teacher/statistics.json','teacher/complete.json','src/models.py','src/prepare_teacher_v3.py']:
        add(name,target=teacher_files)
    main=read(root/'confirmation_summary.json') if (root/'confirmation_summary.json').exists() else {}
    expected=set(job['run_id'] for job in read(root/'confirmation_plan_frozen.json')['jobs']) if (root/'confirmation_plan_frozen.json').exists() else set()
    selected=read(root/'selection_frozen.json') if (root/'selection_frozen.json').exists() else {}
    if selected and len(phase_points.get('validation',[]))==10:
        source=root/'src/evaluate_v3.py'
        require(sha(source)==approval['sources']['evaluate_v3.py'],'Selector source changed before import')
        spec=importlib.util.spec_from_file_location('_verified_v3_pure_selector',source)
        selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
        recomputed,_=selector.select_validation_points(phase_points['validation'])
        require(recomputed==selected['selected'],'Selected settings differ from all10 actual points')
    refs={'B'}|({'C0_matched'} if selected and selected['selected']['C']['kappa']==1. else set())
    if (root/'confirmation_plan_frozen.json').exists():
        confirmation_plan=read(root/'confirmation_plan_frozen.json')
        require(selected and confirmation_plan['selection_sha256']==sha(root/'selection_frozen.json'),
            'Confirmation plan/selection binding differs')
        methods={'B','C'}|({'C0_matched'} if selected['selected']['C']['kappa']==1. else set())
        required_ids={name+'_seed'+str(seed) for name in methods for seed in [101,202,303]}
        jobs=confirmation_plan['jobs']
        require(len(jobs)==len(required_ids) and {job['run_id'] for job in jobs}==required_ids,
            'Frozen plan is not exact fresh3seed selected-method cohort')
        expected=required_ids
    val=phase_records['validation'];test=phase_records['test']
    boot=test.get('bootstrap',{})
    lookup={point['run_id']:point for point in phase_points.get('test',[])}
    for reference,item in boot.items():
        require(reference in refs,'Unplanned primary comparison')
        report_path=owned(root/'test_evaluation',item['path']);report=read(report_path)
        require(report['status']==item['status'],'CI file/index status differs')
        comparison=report['comparison']
        require(comparison['candidate']=='C' and comparison['reference']==reference
            and comparison['fixed_training_seeds']==[101,202,303], 'CI fixedmodel comparison differs')
        candidate_points=[lookup['C_seed'+str(seed)] for seed in [101,202,303]]
        reference_points=[lookup[reference+'_seed'+str(seed)] for seed in [101,202,303]]
        require(comparison['candidate_EMA_sha256']==[p['EMA_sha256'] for p in candidate_points]
            and comparison['reference_EMA_sha256']==[p['EMA_sha256'] for p in reference_points]
            and report['real_feature_sha256']==candidate_points[0]['real_feature_sha256']
            and report['feature_definition']==candidate_points[0]['feature_definition'],
            'CI model/realfeature/extractor binding differs')
        if report['status']=='computed':
            require(report['replicates']==2000 and report['seed']==95003 and report['batch_size']==32
                and report['confidence_one_sided']==.9875 and report['fixed_seed_pairs']==3
                and report['seed_resampling'] is False and report['fake_latent_indices']==512,
                'CI prospective settings differ')
            differences=[]
            for group in range(4):
                key=str(group);a=candidate_points[0]['kid']['groups'][key]
                for point in candidate_points+reference_points:
                    b=point['kid']['groups'][key]
                    require(b['real_ids']==a['real_ids'] and b['fake_requested_ids']==a['fake_requested_ids'],
                        'CI paired point IDs differ')
                require(report['selected_ids'][key]=={'real':a['real_ids'],'fake':a['fake_requested_ids']},
                    'CI selectedIDs differ from actual point estimator')
                difference=sum(a['kid']['groups'][key]['kid']-b['kid']['groups'][key]['kid']
                    for a,b in zip(candidate_points,reference_points))/3
                require(math.isclose(report['groups'][key]['point_difference'],difference,rel_tol=1e-9,abs_tol=1e-10),
                    'CI point difference does not equal mean3 actualseed differences')
                differences.append(difference)
            require(math.isclose(report['rare_group_primary']['point_difference'],differences[3],rel_tol=1e-9,abs_tol=1e-10)
                and math.isclose(report['macro_descriptive']['point_difference'],sum(differences)/4,rel_tol=1e-9,abs_tol=1e-10),
                'CI rare/macro point differs')
    full=complete_report(expected,main,val,test,refs)
    state={'status':'COMPLETE' if full else 'PARTIAL_training_or_points_or_uncertainty',
        'study_id':'generative_recovery_1500','created_utc':datetime.now(timezone.utc).isoformat(),
        'study_plan_sha256':sha(root/'study_plan_frozen.json'),'data_ready_sha256':sha(root/'data/data_ready.json'),
        'main_steps':read(root/'budget_frozen_before_validation.json')['main_steps'],
        'selected_settings':selected.get('selected',{}),
        'fresh_split_group_image_counts':ready['split_group_image_counts'],
        'planned_confirmation_models':len(expected),'complete_confirmation_models':main.get('complete_runs',0),
        'validation_point_count':val.get('point_count',0),'test_point_count':test.get('point_count',0),
        'bootstrap_status':{k:v['status'] for k,v in boot.items()},'expected_primary_references':sorted(refs),
        'one_sided_confidence':.9875,'fixed_training_seeds':[101,202,303],'seed_population_inference':False,
        'feature_NPZ_scope':'exact_producer_selected_real_reference_banks_only_fake_banks_not_exported',
        'selected_reference_union_labels_and_original_rows_explicitly_authorized_for_scientific_replay':True,
        'full_N_private_component_map_or_raw_identity_labels_exported':False,
        'cloud_archives_are_not_proof_of_local_delivery':True,'GPU_or_model_calls_by_exporter':0}
    return files,features,teacher_files,frozen,state


def build(args):
    root=args.root;require(root==ROOT and root.resolve()==ROOT and not root.is_symlink(),'Exact V3 root required')
    approval=read(root/'export_sources_approved.json')
    require(approval['status']=='approved_before_v3_compact_research_export' and approval['exporter_sha256']==sha(__file__)
        and approval['portable_replay_sha256']==sha(root/'runtime/replay_main_ema_v3.py')
        and approval['generic_helper_sha256']==GENERIC_SHA and approval['chunk_max_encoded_bytes']==args.chunk_bytes
        and approval['verifier_sha256']==sha(root/'runtime/verify_v2_transfer.py')
        and approval['training_and_evaluation_processes_terminal'] is True
        and approval['private_derived_split_support_authorized'] is True,'Source/transport/terminal/private-support approval required')
    helper=generic(root);core,banks,teacher,frozen,state=collect(root)
    output=root/'exports_compact';require(not output.exists() and not output.is_symlink(),'Preserve prior export; no retry')
    output.mkdir();stage=output/'staging';stage.mkdir()
    write(stage/'RESEARCH_STATUS.json',state);(stage/'LICENSE_NOTICE.txt').write_text(NOTICE)
    common={'RESEARCH_STATUS.json':stage/'RESEARCH_STATUS.json','LICENSE_NOTICE.txt':stage/'LICENSE_NOTICE.txt'}
    export={**state,'exporter_sha256':sha(__file__),'portable_replay_sha256':sha(root/'runtime/replay_main_ema_v3.py'),
        'ASCII_chunk_max_encoded_bytes':args.chunk_bytes,'export_build_status':'BUILDING','archives':{},
        'all_fake_feature_archives_built':False,'lost_V2_models_or_results_pooled':False}
    write(output/'STARTED.json',export)
    def archive(name,payload,extras=None):
        path=output/name;record=helper.make_archive(path,{**payload,**(common if extras is None else extras)},
            {'study_id':state['study_id'],'status':state['status'],'role':name,'study_plan_sha256':state['study_plan_sha256']})
        chunks=armour(path,output,args.chunk_bytes)
        record['armour_manifest']=chunks.relative_to(output).as_posix();record['armour_manifest_sha256']=sha(chunks)
        export['archives'][name]=record
        write(output/(name+'.complete.json'),{'status':'ARCHIVE_READY_NOT_A_CLAIM_OF_LOCAL_DELIVERY',
            'scientific_status':state['status'],'archives':{name:record}})
        print(json.dumps({'phase':'compact_archive_ready','archive':name,'bytes':record['bytes'],'sha256':record['sha256']}),flush=True)
        return record
    try:
        core_record=archive('research_core.zip',core)
        references={phase+'_evaluation/'+name:banks[phase+'_evaluation/'+name]
            for phase in ['validation','test'] for name in ['real_features.npz','real_features.npz.json','shared_latents.npy']}
        reference={'status':'EXACT_SELECTED_REAL_REFERENCE_AND_SHARED_LATENT_BANKS','study_id':state['study_id'],
            'core_archive':core_record,'study_plan_sha256':state['study_plan_sha256'],
            'data_ready_sha256':state['data_ready_sha256'],'selection_sha256':sha(root/'selection_frozen.json'),
            'files':{name:{'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in references.items()},
            'producer_commit_record':{phase:{'path':phase+'_evaluation/'+('artifact_registry.json'
                if (root/(phase+'_evaluation')/'artifact_registry.json').exists() else 'committed_artifacts.jsonl'),
                'sha256':sha(root/(phase+'_evaluation')/('artifact_registry.json'
                    if (root/(phase+'_evaluation')/'artifact_registry.json').exists() else 'committed_artifacts.jsonl'))}
                for phase in ['validation','test']},
            'fake_feature_banks_included':False,'fullDEV10_regeneration_from_selected_MAIN_weights':False,
            'raw_face_pixels_raw_identity_or_fullN_components_in_this_archive':False}
        write(stage/'REFERENCE_RECORD.json',reference);references['REFERENCE_RECORD.json']=stage/'REFERENCE_RECORD.json'
        archive('replay_references.zip',references)
        archive('teacher_snapshot.zip',teacher)
        ready=read(root/'data/data_ready.json');split={}
        for name,key in [('clean_index.npz','clean_index_sha256'),('component_ids.npy','component_ids_sha256'),
            ('data_ready.json',None),('split_policy.json','split_policy_sha256'),('split_report.json','split_report_sha256')]:
            path=regular(owned(root,'data/'+name))
            require(key is None or sha(path)==ready[key],'Derived split source changed')
            split['data/'+name]=path;frozen[str(path)]=sha(path)
        split_status={**state};split_status.pop('full_N_private_component_map_or_raw_identity_labels_exported',None)
        split_status.update({'archive_scope':'PRIVATE_DERIVED_COMPUTATIONAL_SPLIT_SUPPORT',
            'fullN_derived_union_component_indices_included':True,'raw_supplied_identity_labels_included':False,
            'future_retraining_requires_new_deadline_and_sourceapproval':True})
        write(stage/'SPLIT_SUPPORT_SCOPE.json',split_status)
        (stage/'SPLIT_NOTICE.txt').write_text('PRIVATE DERIVED RESEARCH SPLIT: full-row clean_index.npz and component_ids.npy; '
            'canonical filenames/annotation bins/split masks and computational union indices. Not raw identity TXT/labels, '
            'persons names, images or near edges. Original CelebA research/redistribution terms apply. '
            'Future training needs exact authenticated pixels/new source approval/new clock; no bitwise GPU guarantee.\n')
        archive('split_support.zip',split,{'RESEARCH_STATUS.json':stage/'SPLIT_SUPPORT_SCOPE.json',
            'SPLIT_SUPPORT_SCOPE.json':stage/'SPLIT_SUPPORT_SCOPE.json','LICENSE_NOTICE.txt':stage/'SPLIT_NOTICE.txt'})
        require(all(sha(path)==digest for path,digest in frozen.items()),'Input artifacts changed during export')
        export['export_build_status']='ALL_LISTED_ARCHIVES_BUILT_NOT_A_CLAIM_OF_LOCAL_DELIVERY'
        write(output/'export_manifest.json',export);print(json.dumps(export),flush=True)
    except BaseException as error:
        signal.alarm(0)
        export['export_build_status']='PARTIAL_EXPORT_BUILD_PRESERVED_NO_RETRY';export['error_type']=type(error).__name__
        if not (output/'export_manifest.json').exists():write(output/'export_manifest.json',export)
        raise

def self_test():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);archive=root/'toy.zip'
        raw=b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(110000))
        with zipfile.ZipFile(archive,'x') as z:z.writestr('opaque.bin',raw)
        for cap in sorted(CAPS):
            folder=root/str(cap);folder.mkdir();manifest=armour(archive,folder,cap);value=read(manifest)
            require(value['chunk_count']>=2,'Multichunk fixture required for every cap')
            decoded=b''.join(base64.b64decode((manifest.parent/item['filename']).read_bytes().replace(b'\n',b''),validate=True)
                for item in value['chunks'])
            require(hashlib.sha256(decoded).hexdigest()==sha(archive),'Whole toy reconstruction mismatch')
        try:armour(archive,root,12345)
        except RuntimeError:pass
        else:raise RuntimeError('Unknown transport cap accepted')
    expected={'B_seed101','C_seed101'}
    main={'status':'complete','planned_runs':2,'complete_runs':2,'runs':[{'run_id':v} for v in expected]}
    val={'status':'COMPLETED','point_count':10}
    test={'status':'COMPLETED','point_count':2,'bootstrap':{'B':{'status':'computed'}},'complete_scientific_report':True}
    require(complete_report(expected,main,val,test,{'B'}),'Complete report fixture')
    require(not complete_report(expected,main,val,{**test,'bootstrap':{}},{'B'}),'MissingCI mislabeled complete')
    require(not complete_report(expected,main,val,{**test,'bootstrap':{'B':{'status':'INCOMPLETE'}}},{'B'}),
        'IncompleteCI mislabeled complete')
    require(not complete_report(expected,main,{**val,'point_count':9},test,{'B'}),'Missingvalpoint mislabeled complete')
    print('PASS 3knowncaps/ASCII/multichunk/wholehash/complete-vs-partial; syntheticopaque bytes only')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--chunk-bytes',type=int,default=1048576)
    parser.add_argument('--max-seconds',type=int,default=600);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();require(sys.flags.optimize==0,'Optimized mode refused')
    require(args.chunk_bytes in CAPS and 30<=args.max_seconds<=900,'Invalid fixed transport/time cap')
    if args.self_test:self_test()
    else:
        def timeout(*_):raise TimeoutError('CPU-only research export cap; preserve partials')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(args.max_seconds)
        try:build(args)
        finally:signal.alarm(0)
