"""NEW bounded declared-near audit and prospective fresh component split.

No dataset/network/model work occurs on import. Run only after an authenticated
first-stage identity/exact audit and a root-approved source/input binding.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/kaggle/working/generative_training_v2')
ORIGINAL=Path('/kaggle/working/generative_final_5000')
INDEX_SHA='21dd7eab4a6bb76f8b64e23b1b3b63494e3423bcb03ea26b5db863317890dca5'
CACHE_SHA='98d39e0febb8c1946aba70eca2ca5cb96a657cdf6075f7ae32fc56d4c6f9bd5c'
IDENTITY_MD5='32bd1bd63d3c78cd57e08160ec5ed1e2'
BANDS=[(0,11),(11,11),(22,11),(33,11),(44,10),(54,10)]
MAX_COMPARISONS=250_000_000
MAX_EDGES=1_000_000
MAX_SQUARED_ERROR=786432
POLICY={
 'name':'celeba-v2-component-split-v1','seed':104729,
 'candidate_images':'only_union_components_wholly_inside_original_training',
 'excluded_components':'touch_old_validation_or_test_OR_contain_exact_or_flip_annotation_contradiction',
 'hash':'SHA256_ASCII_celeba-v2-component-split-v1:104729:<minimum_original_zero_based_row>_integer_modulo1000',
 'buckets':{'train':[0,700],'validation':[700,850],'test':[850,1000]},
 'minimum_heldout_images_per_group':100,'minimum_components_per_group_per_split':2,
 'no_partition_search':True,'no_silent_floor_relaxation':True,
 'near_relation':{'phash_bits':64,'maximum_hamming_distance':4,
  'aligned_RGB64_maximum_squared_error':MAX_SQUARED_ERROR,'orientations':['original','horizontal_flip'],
  'candidate_generator':'all15pairs_of6disjoint_bands_11_11_11_11_10_10_complete_by_pigeonhole'},
 'caps':{'inner_wall_seconds':600,'orientation_pair_checks':MAX_COMPARISONS,'accepted_image_pair_edges':MAX_EDGES}}


class AuditError(RuntimeError):pass

def require(condition,message):
    if not condition:raise AuditError(message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):result.update(block)
    return result.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def owned(root,value):
    value=Path(value)
    if not value.is_absolute():value=root/value
    require('..' not in value.parts and not value.is_symlink(),'unsafe_owned_path')
    value=value.resolve();require(value.is_relative_to(root),'path_escaped_owned_root')
    return value


class DisjointSet:
    def __init__(self,n):
        import numpy as np
        self.parent=np.arange(n,dtype=np.int64);self.rank=np.zeros(n,dtype=np.uint8)
    def find(self,value):
        value=int(value)
        while self.parent[value]!=value:
            self.parent[value]=self.parent[self.parent[value]];value=int(self.parent[value])
        return value
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return
        if self.rank[a]<self.rank[b]:a,b=b,a
        self.parent[b]=a
        if self.rank[a]==self.rank[b]:self.rank[a]+=1
    def add_classes(self,labels):
        first={}
        for row,label in enumerate(labels):
            label=int(label)
            if label in first:self.union(row,first[label])
            else:first[label]=row
    def canonical_ids(self):
        import numpy as np
        roots=np.asarray([self.find(i) for i in range(len(self.parent))],dtype=np.int64)
        smallest=np.full(len(roots),len(roots),dtype=np.int64)
        np.minimum.at(smallest,roots,np.arange(len(roots),dtype=np.int64))
        return smallest[roots]


def perceptual_hashes(cache,progress=None):
    import numpy as np
    from PIL import Image
    import scipy.fftpack
    hashes=np.empty(len(cache)*2,dtype=np.uint64)
    for start in range(0,len(cache),512):
        end=min(start+512,len(cache));gray=[]
        for row in range(start,end):
            rgb=np.ascontiguousarray(cache[row].transpose(1,2,0))
            for orientation in [rgb,rgb[:,::-1,:]]:
                image=Image.fromarray(orientation).convert('L').resize((32,32),Image.Resampling.LANCZOS)
                gray.append(np.asarray(image,dtype=np.float64))
        values=np.stack(gray)
        transformed=scipy.fftpack.dct(scipy.fftpack.dct(values,axis=1),axis=2)[:,:8,:8].reshape(-1,64)
        bits=transformed>np.median(transformed,axis=1,keepdims=True)
        packed=np.packbits(bits,axis=1,bitorder='big')
        hashes[2*start:2*end]=packed.copy().view('>u8').reshape(-1).astype(np.uint64)
        if progress and (end%8192==0 or end==len(cache)):progress('phash_images',end,len(cache))
    return hashes


def band_indexes(hashes):
    import numpy as np
    # Each <=4-bit difference touches at most4 of6 disjoint bands, so at least
    # TWO entire bands match. All15 pairs therefore give complete candidates.
    values=np.asarray(hashes,dtype=np.uint64)
    bands=[(values>>shift)&np.uint64((1<<width)-1) for shift,width in BANDS]
    indexes=[]
    for a in range(len(BANDS)):
        for b in range(a+1,len(BANDS)):
            keys=bands[a]|(bands[b]<<BANDS[a][1])
            order=np.argsort(keys,kind='stable')
            sorted_keys=keys[order]
            lower=np.searchsorted(sorted_keys,keys,side='left').astype(np.int32)
            upper=np.searchsorted(sorted_keys,keys,side='right').astype(np.int32)
            indexes.append((order.astype(np.int32),lower,upper))
    return indexes


def candidate_pairs(hashes):
    indexes=band_indexes(hashes)
    for row in range(len(hashes)):
        candidates=set()
        for order,lower,upper in indexes:
            candidates.update(order[lower[row]:upper[row]].tolist())
        for other in sorted(candidates):
            if other>row and other//2!=row//2:yield row,other


def squared_error(cache,a,b):
    import numpy as np
    image_a=cache[a//2];image_b=cache[b//2]
    if a%2:image_a=image_a[:,:,::-1]
    if b%2:image_b=image_b[:,:,::-1]
    difference=image_a.astype(np.int32)-image_b.astype(np.int32)
    return int(np.sum(difference*difference,dtype=np.int64))


def near_edges(cache,hashes,dsu,progress=None,max_comparisons=MAX_COMPARISONS,max_edges=MAX_EDGES):
    checks=0;hamming_pass=0;pixel_checks=0;edges=set()
    for a,b in candidate_pairs(hashes):
        checks+=1;require(checks<=max_comparisons,'UNKNOWN_orientation_pair_cap')
        if (int(hashes[a])^int(hashes[b])).bit_count()<=4:
            hamming_pass+=1;pair=tuple(sorted((a//2,b//2)))
            if pair not in edges:
                pixel_checks+=1
                if squared_error(cache,a,b)<=MAX_SQUARED_ERROR:
                    require(len(edges)<max_edges,'UNKNOWN_accepted_edge_cap')
                    edges.add(pair);dsu.union(*pair)
        if progress and checks%500_000==0:progress('candidate_orientation_pairs',checks,max_comparisons)
    return edges,{'status':'COMPLETED_declared_relation_screen','orientation_pair_checks':checks,
        'hamming_pass_orientation_pairs':hamming_pass,'full_RGB_pixel_checks':pixel_checks,
        'accepted_unique_filename_pair_edges':len(edges),'complete_candidate_generation':True,
        'candidate_generator':'all15_pairs_of6_disjoint_bands_sorted_numpy_indexes','band_widths':[11,11,11,11,10,10],
        'maximum_hamming_distance':4,'maximum_RGB_squared_error':MAX_SQUARED_ERROR,
        'all_canonical_images_and_horizontal_reflections_included':True,
        'universal_semantic_duplicate_guarantee':False}


def contradictory_component_ids(exact_flip_labels,groups,union_ids):
    first={};contradictory=set()
    for row,label in enumerate(exact_flip_labels):
        label=int(label);group=int(groups[row])
        if label in first and first[label][0]!=group:
            contradictory.add(int(union_ids[row]))
        else:first.setdefault(label,(group,row))
    return contradictory


def frozen_partition(original_splits,groups,union_ids,exact_flip_labels,minimum_images=100,minimum_components=2):
    import numpy as np
    require(len(original_splits)==len(groups)==len(union_ids)==len(exact_flip_labels),'split_array_lengths')
    contradiction=contradictory_component_ids(exact_flip_labels,groups,union_ids)
    touches_old_heldout=set(int(c) for c,s in zip(union_ids,original_splits) if s!=0)
    all_ids=set(map(int,union_ids));eligible=all_ids-touches_old_heldout-contradiction
    assignment={}
    for component in sorted(eligible):
        text=f'celeba-v2-component-split-v1:104729:{component}'.encode('ascii')
        bucket=int.from_bytes(hashlib.sha256(text).digest(),'big')%1000
        assignment[component]=0 if bucket<700 else (1 if bucket<850 else 2)
    splits=np.asarray([assignment.get(int(component),-1) for component in union_ids],dtype=np.int64)
    require(not np.any((splits>=0)&(original_splits!=0)),'original_heldout_image_would_be_used')
    require(all(int(s)<0 or assignment[int(c)]==int(s) for c,s in zip(union_ids,splits)),
        'component_crossed_new_split')
    counts=[[int(np.sum((splits==s)&(groups==g))) for g in range(4)] for s in range(3)]
    component_counts=[[len(set(map(int,union_ids[(splits==s)&(groups==g)]))) for g in range(4)] for s in range(3)]
    require(all(value>=minimum_images for row in counts[1:] for value in row),'heldout_group_image_floor_failed_no_partition_search')
    require(all(value>=minimum_components for row in component_counts for value in row),'group_component_floor_failed_no_partition_search')
    return splits,{'split_group_image_counts':counts,'split_group_component_counts':component_counts,
        'eligible_components':len(eligible),'excluded_touching_old_validation_or_test_components':len(touches_old_heldout),
        'exact_or_flip_annotation_conflict_union_components':len(contradiction),
        'quarantined_rows':int(np.count_nonzero(splits<0)),'rows_quarantined_due_to_exact_or_flip_conflict':sum(int(c) in contradiction for c in union_ids),
        'new_split_total_images':[int(np.sum(splits==s)) for s in range(3)],
        'original_validation_or_test_images_used':0,'component_assignment':'fixed_hash_70_15_15',
        'partition_search_performed':False,'all_split_component_intersections':0}


def main(args):
    require(sys.flags.optimize==0,'optimized_mode_not_allowed')
    require(ROOT.resolve()==ROOT and ORIGINAL.resolve()==ORIGINAL and not ROOT.is_symlink()
        and not ORIGINAL.is_symlink(),'exact_owned_roots_required')
    approval_path=owned(ROOT,args.approval);approval=read(approval_path)
    require(approval['status']=='approved_before_v2_near_and_fresh_split'
        and approval['worker_sha256']==sha(__file__),'source_approval_required')
    timestamp=datetime.fromisoformat(approval['approved_utc'].replace('Z','+00:00'))
    require(timestamp.tzinfo is not None and timestamp<=datetime.now(timezone.utc),'approval_timestamp')
    audit=owned(ROOT,approval['first_stage_directory']);output=owned(ROOT,args.output)
    require(output==ROOT/'data' and not output.exists() and not output.is_symlink(),'fresh_v2_data_output_required')
    require(sha(audit/'summary.json')==approval['first_stage_summary_sha256'],'first_stage_summary_binding')
    summary=read(audit/'summary.json')
    require(summary['status']=='completed_v2_narrow_data_audit' and summary['inputs_unchanged'] is True
        and summary['identity_status']=='completed_supplied_identity_label_audit'
        and summary['source_sha256']==approval['first_stage_worker_sha256'],'authenticated_first_stage_required')
    require(approval['first_stage_worker_path'] in ['audit/audit_exact_identity.py','audit/audit_exact_identity_recovery.py']
        and sha(owned(ROOT,approval['first_stage_worker_path']))==approval['first_stage_worker_sha256'],'first_stage_source_changed')
    first_approval=owned(ROOT,approval['first_stage_approval_path'])
    require(sha(first_approval)==summary['approval_sha256'],'first_stage_approval_changed')
    for name,digest in summary['public_report_files'].items():
        require(Path(name).name==name and sha(audit/name)==digest,'first_stage_public_record_changed')
    identity_path=audit/'private/identity_labels.npy';components_path=audit/'private/exact_component_labels.npz'
    require(sha(identity_path)==approval['identity_labels_sha256']
        and sha(components_path)==approval['exact_component_labels_sha256'],'private_audit_arrays_binding')
    original_protocol=ORIGINAL/'protocol_frozen.json'
    require(sha(original_protocol)==approval['original_protocol_sha256'],'original_protocol_changed')
    original=read(original_protocol)
    require(original['study_id']=='generative_final_5000','wrong_original_study')
    index=owned(ORIGINAL,original['training_config']['metadata_npz'])
    cache_path=owned(ORIGINAL,original['training_config']['cache_path'])
    output.mkdir()
    write(output/'started.json',{'status':'started_new_near_screen_and_fresh_split','utc':utc(),
        'source_sha256':sha(__file__),'approval_sha256':sha(approval_path),'test_quality_accessed':False,
        'inner_cap_seconds':600,'policy':POLICY})
    began=time.monotonic();old=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise AuditError('UNKNOWN_600second_phase_cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,600)
    before={}
    try:
        import numpy as np
        import scipy
        import PIL
        before={str(path):sha(path) for path in [index,cache_path,original_protocol,approval_path,
            identity_path,components_path,audit/'summary.json',Path(__file__),
            owned(ROOT,approval['first_stage_worker_path']),first_approval]}
        require(before[str(index)]==INDEX_SHA and before[str(cache_path)]==CACHE_SHA,'canonical_inputs_changed')
        with np.load(index,allow_pickle=False) as values:
            names=values['filenames'];groups=values['groups'];original_splits=values['splits']
        identity=np.load(identity_path,allow_pickle=False)
        with np.load(components_path,allow_pickle=False) as values:
            direct=values['direct_component'];flipped=values['horizontal_flip_component']
        require(names.shape==groups.shape==original_splits.shape==identity.shape==direct.shape==flipped.shape==(202599,),
            'audit_array_shape_mismatch')
        require(identity.dtype.kind in 'iu' and direct.dtype.kind in 'iu' and flipped.dtype.kind in 'iu'
            and bool((identity>0).all()) and bool(((direct>=0)&(direct<len(names))).all())
            and bool(((flipped>=0)&(flipped<len(names))).all()) and len(np.unique(identity))==10177,'identity_or_component_array_invalid')
        cache=np.load(cache_path,mmap_mode='r',allow_pickle=False)
        require(cache.shape==(202599,3,64,64) and cache.dtype==np.uint8,'cache_shape_type')
        dsu=DisjointSet(len(names));dsu.add_classes(identity);dsu.add_classes(direct);dsu.add_classes(flipped)
        def progress(phase,count,total):print(json.dumps({'phase':phase,'completed':count,'cap_or_total':total}),flush=True)
        hashes=perceptual_hashes(cache,progress)
        np.save(output/'private_phash64_orientations.npy',hashes,allow_pickle=False)
        edges,near=near_edges(cache,hashes,dsu,progress)
        edge_array=np.asarray(sorted(edges),dtype=np.int64).reshape(-1,2)
        np.save(output/'private_near_edges.npy',edge_array,allow_pickle=False)
        near.update({'source_sha256':sha(__file__),'numpy':np.__version__,'scipy':scipy.__version__,
            'Pillow':PIL.__version__,'elapsed_seconds':time.monotonic()-began,
            'descriptor':'PIL_L32_LANCZOS_scipy_fftpack_DCT_axes12_upper8x8_gt_median64_big_endian_bits',
            'private_descriptor_sha256':sha(output/'private_phash64_orientations.npy'),
            'private_edge_array_sha256':sha(output/'private_near_edges.npy')})
        write(output/'near_screen_summary.json',near)
        union_ids=dsu.canonical_ids()
        clean_splits,split_report=frozen_partition(original_splits,groups,union_ids,flipped)
        np.save(output/'component_ids.npy',union_ids,allow_pickle=False)
        np.savez_compressed(output/'clean_index.npz',filenames=names,groups=groups,splits=clean_splits)
        write(output/'split_policy.json',POLICY)
        write(output/'split_report.json',split_report)
        after={name:sha(name) for name in before};require(after==before,'original_inputs_changed_during_phase2')
        ready={'status':'frozen_clean_component_split','study_id':'generative_training_v2',
            'utc':utc(),'source_sha256':sha(__file__),'approval_sha256':sha(approval_path),
            'original_index_sha256':INDEX_SHA,'original_cache_sha256':CACHE_SHA,
            'cache_sha256':CACHE_SHA,'cache_path':str(cache_path),
            'metadata_npz':str(output/'clean_index.npz'),'clean_index_sha256':sha(output/'clean_index.npz'),
            'component_ids_path':str(output/'component_ids.npy'),'component_ids_sha256':sha(output/'component_ids.npy'),
            'split_policy_sha256':sha(output/'split_policy.json'),'audit_summary_sha256':sha(audit/'summary.json'),
            'near_screen_summary_sha256':sha(output/'near_screen_summary.json'),
            'split_report_sha256':sha(output/'split_report.json'),**split_report,
            'preparation_source_path':'audit/build_clean_component_split.py',
            'identity_label_audit_status':'completed_authenticated_MD5_and_filename_join',
            'declared_near_relation_completed':True,'test_quality_accessed':False,
            'old_official_test_quality_used_for_selection':False,'original_val_test_images_used':False,
            'all_v2_teachers_candidates_and_comparators_must_train_from_scratch':True,
            'scope':'disjoint_under_supplied_identity_exact_flip_and_declared_near_relation_not_all_semantic_similarity',
            'inputs_before':before,'inputs_after':after,'inputs_unchanged':True,
            'elapsed_seconds':time.monotonic()-began,'private_raw_identity_or_face_images_exported':False}
        write(output/'data_ready.json',ready)
        print(json.dumps({'status':ready['status'],'data_ready_sha256':sha(output/'data_ready.json'),
            'split_group_image_counts':ready['split_group_image_counts'],
            'split_group_component_counts':ready['split_group_component_counts']}),flush=True)
    except BaseException as error:
        write(output/'failure.json',{'status':'UNKNOWN_or_INCOMPLETE_near_screen_or_split','utc':utc(),
            'error_type':type(error).__name__,'error_code':str(error) if isinstance(error,AuditError) else 'sanitized_unexpected_error',
            'partial_records_preserved':True,'no_training_gate_opened':True,'no_automatic_retry':True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def self_test():
    import numpy as np
    # Literal descriptors: pigeonhole must retain every true distance<=4 pair.
    hashes=np.asarray([0,0,1|(1<<13)|(1<<26)|(1<<39),31,0xffffffffffffffff,
        0xfffffffffffffff0,1<<63,3<<25,1<<52,1<<51],dtype=np.uint64)
    candidates=set(candidate_pairs(hashes))
    for a in range(len(hashes)):
        for b in range(a+1,len(hashes)):
            if a//2!=b//2 and (int(hashes[a])^int(hashes[b])).bit_count()<=4:
                require((a,b) in candidates,'Hamming_radius_candidate_completeness_fixture')
    require(all(a//2!=b//2 for a,b in candidates),'self_filename_pair_not_removed')
    black=np.zeros((3,64,64),dtype=np.uint8)
    white=np.full_like(black,255)
    cache=np.stack([black,white,np.full_like(black,8),np.full_like(black,9)])
    require(squared_error(cache,0,2)==255*255*12288,'unsigned_underflow_fixture')
    require(squared_error(cache,0,4)==MAX_SQUARED_ERROR and squared_error(cache,0,6)>MAX_SQUARED_ERROR,'pixel_threshold_fixture')
    dsu=DisjointSet(4)
    edges,report=near_edges(cache,np.zeros(8,dtype=np.uint64),dsu)
    require((0,2) in edges and (0,1) not in edges and report['complete_candidate_generation'],'pixel_filtered_edges_fixture')
    try:near_edges(cache,np.zeros(8,dtype=np.uint64),DisjointSet(4),max_comparisons=0)
    except AuditError as error:require('UNKNOWN' in str(error),'cap_unknown_fixture')
    else:raise AuditError('cap_not_rejected')
    dsu=DisjointSet(8);dsu.add_classes([1,1,2,3,4,5,6,6]);dsu.union(1,2)
    ids=dsu.canonical_ids();require(ids.tolist()==[0,0,0,3,4,5,6,6],'union_canonical_transitivity_fixture')
    original=np.asarray([0,0,1,0,0,0,0,0]);groups=np.asarray([0,1,0,1,2,3,2,3]);exact=np.arange(8);exact[7]=6
    new,report=frozen_partition(original,groups,ids,exact,minimum_images=0,minimum_components=0)
    require(np.all(new[:3]==-1) and np.all(new[6:]==-1),'touch_heldout_and_conflict_quarantine_fixture')
    require(report['original_validation_or_test_images_used']==0,'original_heldout_exclusion_fixture')
    try:frozen_partition(original,groups,ids,exact)
    except AuditError:pass
    else:raise AuditError('support_floor_not_enforced')
    # Actual descriptor construction toy only, no real cache or learned model.
    descriptors=perceptual_hashes(np.stack([black,white]))
    require(descriptors.shape==(4,) and descriptors[0]==descriptors[1] and descriptors[2]==descriptors[3],
        'descriptor_shape_and_symmetric_flip_fixture')
    print('PASS toy complete-band-candidates/thresholds/caps/union/quarantine/floors/descriptor; no actual dataset or quality')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval',type=Path,default=ROOT/'near_split_sources_approved.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data')
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    else:main(args)
