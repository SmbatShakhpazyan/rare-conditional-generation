"""NEW recovery version: native-JSON exact/flip audit and optional authenticated identity INPUT.

Preserves failed attempt001 and its source. NO HTTP/network/download path.
A new attempt requires source-bound recovery approval; identity is UNKNOWN unless
root explicitly supplies an owned file with the official MD5 and approved SHA.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

INPUT_ROOT=Path('/kaggle/working/generative_final_5000')
V2_ROOT=Path('/kaggle/working/generative_training_v2')
INDEX_SHA='21dd7eab4a6bb76f8b64e23b1b3b63494e3423bcb03ea26b5db863317890dca5'
CACHE_SHA='98d39e0febb8c1946aba70eca2ca5cb96a657cdf6075f7ae32fc56d4c6f9bd5c'
IDENTITY_MD5='32bd1bd63d3c78cd57e08160ec5ed1e2'
COUNTS=[[71629,22880,66874,1387],[8535,2874,8276,182],[9767,2480,7535,180]]
PAIRS=[(0,1),(0,2),(1,2)]
SPLITS=['train','validation','test']


class AuditError(RuntimeError):pass

def require(value,message):
    if not value:raise AuditError(message)


def sha(path,algorithm='sha256'):
    digest=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def owned(root,value):
    path=Path(value)
    if not path.is_absolute():path=root/path
    require('..' not in path.parts and not path.is_symlink(),'unsafe_owned_path')
    path=path.resolve();require(path.is_relative_to(root),'path_escaped_owned_root')
    return path


def gate(args):
    require(args.input_root==INPUT_ROOT and args.v2_root==V2_ROOT
        and not args.input_root.is_symlink() and not args.v2_root.is_symlink()
        and args.input_root.resolve()==INPUT_ROOT and args.v2_root.resolve()==V2_ROOT,'exact_owned_roots_required')
    approval_path=owned(V2_ROOT,args.approval);approval=read(approval_path)
    require(approval['status']=='approved_before_v2_exact_identity_recovery'
        and approval['worker_sha256']==sha(__file__)
        and approval['input_root']==str(INPUT_ROOT)
        and approval['index_sha256']==INDEX_SHA and approval['cache_sha256']==CACHE_SHA,'source_bound_audit_approval_required')
    stamp=datetime.fromisoformat(approval['approved_utc'].replace('Z','+00:00'))
    require(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),'approval_timestamp_invalid')
    require(approval['previous_worker_sha256']=='6cbc4fe3cedcd6b917363edc2e1b9bb029c632d590279c51890a0d6e3897c28e'
        and sha(V2_ROOT/'audit/audit_exact_identity.py')==approval['previous_worker_sha256'],'preserved_previous_source_required')
    previous=owned(V2_ROOT,approval['previous_attempt_directory'])
    require(previous==V2_ROOT/'audit/attempt001' and sha(previous/'failure.json')==approval['previous_failure_sha256']
        and read(previous/'failure.json')['status']=='INCOMPLETE_v2_audit','actual_preserved_failure_required')
    require(sha(INPUT_ROOT/'protocol_frozen.json')==approval['input_protocol_sha256'],'input_protocol_changed')
    output=owned(V2_ROOT,args.output)
    require(output!=V2_ROOT and output.parent!=V2_ROOT,'nested_attempt_output_required')
    return approval,approval_path,output


def parse_identities(path,names):
    expected=set(names);joined={}
    with Path(path).open(encoding='ascii') as f:
        for line in f:
            fields=line.split();require(len(fields)==2,'identity_bad_field_count')
            name,label=fields
            require(name in expected and name not in joined,'identity_bad_or_duplicate_filename')
            require(re.fullmatch('[1-9][0-9]*',label) is not None,'identity_not_positive_integer')
            joined[name]=int(label)
    require(len(joined)==len(names),'identity_incomplete_filename_coverage')
    return [joined[name] for name in names]


def identity_summary(labels,splits,groups):
    ids=[set(label for label,split in zip(labels,splits) if split==s) for s in range(3)]
    by_split={SPLITS[s]:{'images':sum(split==s for split in splits),'unique_identity_labels':len(ids[s]),
        'groups':{str(g):len({labels[i] for i in range(len(labels)) if splits[i]==s and groups[i]==g}) for g in range(4)}} for s in range(3)}
    overlap={SPLITS[a]+'__'+SPLITS[b]:len(ids[a]&ids[b]) for a,b in PAIRS}
    return {'status':'completed_supplied_identity_label_audit','distinct_supplied_labels':len(set(labels)),
        'pairwise_identity_label_intersections':overlap,'all_three_identity_label_intersection':len(ids[0]&ids[1]&ids[2]),
        'by_split':by_split,'validation_images_with_training_identity_label':sum(s==1 and x in ids[0] for x,s in zip(labels,splits)),
        'test_images_with_training_identity_label':sum(s==2 and x in ids[0] for x,s in zip(labels,splits)),
        'no_identity_label_overlap':not any(overlap.values()),'real_person_identity_truth_certified':False}


def exposure_summary(exposure,groups,splits,expected_total=320000,expected_groups=(80000,)*4):
    import numpy as np
    require(exposure.shape==(len(groups),) and exposure.dtype.kind in 'iu' and np.all(exposure>=0),'invalid_exposure_vector')
    counts=[int(exposure[splits==s].sum(dtype=np.int64)) for s in range(3)]
    group_counts=[int(exposure[groups==g].sum(dtype=np.int64)) for g in range(4)]
    require(sum(counts)==expected_total and group_counts==list(expected_groups),'exposure_totals_differ')
    require(counts[1:]==[0,0] and np.count_nonzero(exposure[splits!=0])==0,'HELDOUT_ROWS_PRESENT_IN_TRAINING_EXPOSURES')
    return {'total_presentations':sum(counts),'presentations_by_split':counts,'group_presentations':group_counts,
        'positive_nontraining_rows':0,'unique_training_images':int(np.count_nonzero(exposure)),
        'group_unique_images':[int(np.count_nonzero(exposure[groups==g])) for g in range(4)]}


def pixel_bytes(image,flip_canonical=False):
    import numpy as np
    rgb=np.ascontiguousarray(image.transpose(1,2,0))
    original=rgb.tobytes()
    return min(original,rgb[:,::-1,:].tobytes()) if flip_canonical else original


def summarize_components(components,groups,splits):
    duplicate=[members for members in components if len(members)>1]
    cross={SPLITS[a]+'__'+SPLITS[b]:0 for a,b in PAIRS};conflicts=dict(cross)
    images_cross=set();conflict_groups=0;male_conflicts=0;blond_conflicts=0
    for members in duplicate:
        sc=Counter(int(splits[i]) for i in members);gc=Counter(int(groups[i]) for i in members)
        if len(sc)>1:images_cross.update(members)
        conflict_groups+=int(len(gc)>1)
        male_conflicts+=int(len({int(groups[i])//2 for i in members})>1)
        blond_conflicts+=int(len({int(groups[i])%2 for i in members})>1)
        for a,b in PAIRS:
            key=SPLITS[a]+'__'+SPLITS[b];total=sc[a]*sc[b]
            same=sum(sum(int(splits[i])==a and int(groups[i])==g for i in members)
                *sum(int(splits[i])==b and int(groups[i])==g for i in members) for g in range(4))
            cross[key]+=total;conflicts[key]+=total-same
    return {'duplicate_components':len(duplicate),'images_in_duplicate_components':sum(map(len,duplicate)),
        'redundant_rows':sum(len(m)-1 for m in duplicate),'cross_split_pairs':cross,
        'images_in_cross_split_components':len(images_cross),'conflicting_annotation_components':conflict_groups,
        'conflicting_Male_components':male_conflicts,'conflicting_Blond_Hair_components':blond_conflicts,
        'cross_split_conflicting_annotation_pairs':conflicts}


def scan_pixels(cache,groups,splits,progress=None,digest_function=None):
    import numpy as np
    digest_function=digest_function or (lambda value:hashlib.sha256(value).digest())
    maps=[{},{}];classes=[[],[]]
    labels=[np.empty(len(cache),dtype=np.int32),np.empty(len(cache),dtype=np.int32)]
    verified_collision_buckets=[0,0]
    for row in range(len(cache)):
        original=pixel_bytes(cache[row]);flipped=min(original,pixel_bytes(cache[row,:, :, ::-1]))
        for mode,value in enumerate([original,flipped]):
            digest=digest_function(value);bucket=maps[mode].setdefault(digest,[]);found=None
            for component in bucket:
                representative=classes[mode][component][0]
                if value==pixel_bytes(cache[representative],bool(mode)):
                    found=component;verified_collision_buckets[mode]+=1;break
            if found is None:
                found=len(classes[mode]);classes[mode].append([]);bucket.append(found)
            classes[mode][found].append(row);labels[mode][row]=found
        if progress and (row+1)%8192==0:progress(row+1)
    summary={'direct':summarize_components(classes[0],groups,splits),
        'horizontal_flip_equivalence':summarize_components(classes[1],groups,splits),
        'hash_collision_candidates_verified_by_actual_bytes':verified_collision_buckets,
        'exact_bytes':'uint8_RGB64_HWC_row_major','horizontal_flip_matches_training_augmentation':True,
        'logical_equality_based_on_bytes_not_digest_alone':True}
    summary['additional_mirror_cross_split_pairs']={key:summary['horizontal_flip_equivalence']['cross_split_pairs'][key]
        -summary['direct']['cross_split_pairs'][key] for key in summary['direct']['cross_split_pairs']}
    require(all(n>=0 for n in summary['additional_mirror_cross_split_pairs'].values()),'mirror_equivalence_refinement_failed')
    return summary,labels


def audit(args):
    approval,approval_path,output=gate(args)
    require(not output.exists() and not output.is_symlink(),'fresh_attempt_directory_required')
    output.mkdir(parents=True);(output/'private').mkdir();began=time.monotonic();before={}
    write(output/'started.json',{'status':'started_v2_read_only_audit','utc':utc(),'worker_sha256':sha(__file__),
        'approval_sha256':sha(approval_path),'no_quality_values_accessed':True,'inner_cap_seconds':420})
    old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise AuditError('audit_420second_inner_cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,420)
    try:
        import numpy as np
        protocol_path=INPUT_ROOT/'protocol_frozen.json';protocol=read(protocol_path)
        require(protocol['study_id']=='generative_final_5000' and protocol['matrix']['steps']==5000,'wrong_input_study')
        index=owned(INPUT_ROOT,protocol['training_config']['metadata_npz']);cache_path=owned(INPUT_ROOT,protocol['training_config']['cache_path'])
        before={str(path):sha(path) for path in [protocol_path,index,cache_path,approval_path,Path(__file__)]}
        require(before[str(index)]==INDEX_SHA and before[str(cache_path)]==CACHE_SHA,'whole_input_byte_identity_failed')
        for name,digest in protocol['source_sha256'].items():
            require(Path(name).name==name and sha(INPUT_ROOT/'src'/name)==digest,'original_source_changed')
            before[str(INPUT_ROOT/'src'/name)]=digest
        with np.load(index,allow_pickle=False) as data:
            require(set(data.files)=={'filenames','groups','splits'},'unexpected_index_arrays')
            names=data['filenames'].astype(str);groups=data['groups'];splits=data['splits']
        require(names.shape==groups.shape==splits.shape==(202599,) and len(set(names.tolist()))==202599
            and names.tolist()==[f'{i:06d}.jpg' for i in range(1,202600)],'filename_coverage_or_order_invalid')
        require(groups.dtype.kind in 'iu' and splits.dtype.kind in 'iu' and np.isin(groups,range(4)).all()
            and np.isin(splits,range(3)).all(),'invalid_index_categories')
        counts=[[int(np.sum((splits==s)&(groups==g))) for g in range(4)] for s in range(3)]
        require(counts==COUNTS,'official_counts_differ')
        require(protocol['matrix']['arms']==['B','P','V2','GSR']
            and protocol['matrix']['training_seeds']==[101,202,303]
            and len(protocol['matrix']['order'])==12
            and {(int(seed),arm) for seed,arm in protocol['matrix']['order']}
                =={(seed,arm) for seed in [101,202,303] for arm in ['B','P','V2','GSR']},'exact_original12_matrix_required')
        exposure_records={}
        for seed,arm in protocol['matrix']['order']:
            key=f'{arm}_seed{seed}';folder=INPUT_ROOT/'runs'/key
            complete_path=folder/'complete.json';exposure_path=folder/'exposures.npy';complete=read(complete_path)
            require(complete['status']=='complete' and complete['completed_steps']==5000
                and complete['arm']==arm and complete['seed']==seed and complete['pilot'] is False
                and complete['real_images']==320000
                and complete['config']==protocol['training_config'],'wrong_actual_training_record')
            before[str(complete_path)]=sha(complete_path);before[str(exposure_path)]=sha(exposure_path)
            vector=np.load(exposure_path,mmap_mode='r',allow_pickle=False)
            record=exposure_summary(vector,groups,splits)
            require(record['group_presentations']==complete['group_exposures']
                and record['group_unique_images']==complete['group_unique_images'],'logged_exposure_accounting_differs')
            exposure_records[key]={**record,'exposures_sha256':before[str(exposure_path)]}
        require(len(exposure_records)==12,'all12_exposure_records_required')
        write(output/'filenames_and_exposures.json',{'status':'PASS_actual_filename_and_exposure_separation',
            'rows':202599,'unique_filenames':202599,'pairwise_filename_intersections':{SPLITS[a]+'__'+SPLITS[b]:0 for a,b in PAIRS},
            'split_group_counts':counts,'runs':exposure_records,'no_identity_or_near_duplicate_claim':True})
        identity_result={'status':'UNKNOWN_not_acquired','no_identity_disjointness_claim':True}
        if args.identity_file is not None:
            identity_file=owned(V2_ROOT,args.identity_file)
            require(sha(identity_file)==approval['identity_file_sha256'] and sha(identity_file,'md5')==IDENTITY_MD5,
                'provided_identity_not_authenticated')
            identity_labels=parse_identities(identity_file,names.tolist())
            identity_result=identity_summary(identity_labels,splits.tolist(),groups.tolist())
            require(identity_result['distinct_supplied_labels']==10177,'author_identity_total_differs')
            identity_result['identity_file_MD5']=IDENTITY_MD5;identity_result['identity_file_sha256']=sha(identity_file)
            identity_result['acquisition_scope']='root_supplied_authenticated_input_no_network_from_recovery_worker'
            np.save(output/'private/identity_labels.npy',np.asarray(identity_labels,dtype=np.int32),allow_pickle=False)
            before[str(identity_file)]=sha(identity_file)
        write(output/'identity_aggregate.json',identity_result)
        cache=np.load(cache_path,mmap_mode='r',allow_pickle=False)
        require(cache.shape==(202599,3,64,64) and cache.dtype==np.uint8,'wrong_RGB64_cache_shape_type')
        pixel_result,component_labels=scan_pixels(cache,groups,splits,
            lambda n:print(json.dumps({'phase':'exact_and_flip_pixels','completed_rows':n,'total_rows':202599}),flush=True))
        np.savez_compressed(output/'private/exact_component_labels.npz',direct_component=component_labels[0],
            horizontal_flip_component=component_labels[1])
        write(output/'exact_pixel_aggregate.json',pixel_result)
        after={name:sha(name) for name in before};require(after==before,'original_inputs_changed_during_audit')
        cross=pixel_result['horizontal_flip_equivalence']['cross_split_pairs']
        all_audited=(identity_result.get('no_identity_label_overlap') is True and not any(cross.values()))
        result={'status':'completed_v2_narrow_data_audit','utc':utc(),'rows':202599,
            'filename_and_actual_exposure_checks':'PASS','exact_and_flip_cross_split_pairs':cross,
            'identity_status':identity_result['status'],'all_completed_disjointness_checks_passed':all_audited,
            'full_strict_gate':'PENDING_NEAR_SCREEN_AND_SPLIT_LOCK','near_duplicate_screen_status':'NOT_RUN',
            'metadata_identity_truth_guarantee':False,'source_sha256':sha(__file__),'approval_sha256':sha(approval_path),
            'inputs_before':before,'inputs_after':after,'inputs_unchanged':True,'numpy':np.__version__,
            'elapsed_seconds':time.monotonic()-began,'original_splits_changed':False,'quality_values_accessed':False,
            'public_report_files':{name:sha(output/name) for name in ['filenames_and_exposures.json','identity_aggregate.json','exact_pixel_aggregate.json']},
            'private_raw_identity_or_image_data_exported':False,
            'limits':['canonicalRGB64andhorizontalreflectionequalityonly','suppliedidentitylabelsnotinfalliblerealpersontruth',
                'noduplicatecleaningorsplitreassignment','near-duplicatesnotyetscreened','source/exposurelogsarenotanindependentprocessmemorytrace']}
        write(output/'summary.json',result);print(json.dumps({'status':result['status'],
            'summary_sha256':sha(output/'summary.json'),'quality_values_accessed':False}),flush=True)
    except BaseException as error:
        write(output/'failure.json',{'status':'INCOMPLETE_v2_audit','utc':utc(),'error_type':type(error).__name__,
            'error_code':str(error) if isinstance(error,AuditError) else 'sanitized_unexpected_error',
            'source_sha256':sha(__file__),'partial_records_preserved':True,'no_automatic_retry':True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)


def self_test():
    import numpy as np
    import tempfile
    a=np.zeros((3,64,64),dtype=np.uint8);a[0,:,:32]=17;a[1,:,32:]=90
    cache=np.stack([a,a,a[:,:,::-1],np.full_like(a,128)])
    groups=np.array([0,3,0,2]);splits=np.array([0,1,2,2])
    summary,labels=scan_pixels(cache,groups,splits)
    json.dumps(summary,allow_nan=False)  # Regression: actual aggregate must contain native JSON values.
    require(summary['direct']['cross_split_pairs']=={'train__validation':1,'train__test':0,'validation__test':0},'direct_fixture')
    require(summary['horizontal_flip_equivalence']['cross_split_pairs']=={'train__validation':1,'train__test':1,'validation__test':1},'flip_fixture')
    require(summary['additional_mirror_cross_split_pairs']=={'train__validation':0,'train__test':1,'validation__test':1},'additional_flip_fixture')
    require(summary['direct']['conflicting_Male_components']==summary['direct']['conflicting_Blond_Hair_components']==1,'annotation_conflict_fixture')
    collided,_=scan_pixels(cache,groups,splits,digest_function=lambda _:b'FORCED_DIGEST_COLLISION')
    json.dumps(collided,allow_nan=False)
    require(collided['direct']==summary['direct'] and collided['horizontal_flip_equivalence']==summary['horizontal_flip_equivalence'],'actual_byte_collision_verification_fixture')
    identity=identity_summary([10,20,10,30],splits.tolist(),groups.tolist())
    require(identity['pairwise_identity_label_intersections']['train__test']==1 and identity['test_images_with_training_identity_label']==1,'identity_overlap_fixture')
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'identities.txt';path.write_text('000002.jpg 20\n000001.jpg 10\n')
        require(parse_identities(path,['000001.jpg','000002.jpg'])==[10,20],'filename_join_fixture')
        path.write_text('000001.jpg 10\n000001.jpg 20\n')
        try:parse_identities(path,['000001.jpg','000002.jpg'])
        except AuditError:pass
        else:raise AuditError('duplicate_identity_not_rejected')
    exposure_summary(np.array([5,0,0,0]),groups,splits,5,(5,0,0,0))
    try:exposure_summary(np.array([4,0,1,0]),groups,splits,5,(5,0,0,0))
    except AuditError:pass
    else:raise AuditError('heldout_exposure_not_rejected')
    print('PASS: exact/flip/conflicts/forceddigestcollision/identityjoin/identityoverlap/exposurefixtures; no network/dataset/models')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=INPUT_ROOT);parser.add_argument('--v2-root',type=Path,default=V2_ROOT)
    parser.add_argument('--approval',type=Path,default=V2_ROOT/'data_audit_recovery_sources_approved.json')
    parser.add_argument('--output',type=Path,default=V2_ROOT/'audit/attempt002')
    parser.add_argument('--identity-file',type=Path,default=None)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    else:audit(args)
