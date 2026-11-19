"""NEW standalone recovery of canonical inputs and the SAME clean split.
Attached public CelebA JPEG/CSV inputs only; one approved university identity GET.
No old-run gates, models, Torch, GPU, quality reads, retries or threshold changes.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import ssl
import subprocess
import sys
import time
import urllib.request
import urllib.parse

ROOT=Path('/kaggle/working/generative_recovery_2500')
ATTACHED=Path('/kaggle/input/datasets/jessicali9530/celeba-dataset')
SCRATCH=Path('/tmp/generative_recovery_2500')
N=202599
EXPECTED={'original_index_sha256':'21dd7eab4a6bb76f8b64e23b1b3b63494e3423bcb03ea26b5db863317890dca5',
 'original_cache_sha256':'98d39e0febb8c1946aba70eca2ca5cb96a657cdf6075f7ae32fc56d4c6f9bd5c',
 'clean_index_sha256':'d278e0a67060e70416314a05cb8ea29d6ccd41b80e22ab90c89226bd847156e1',
 'component_ids_sha256':'bb039432c9bc102fb7d0555a8d381293671d46c46a62bf624dcc0aab31239e8d'}
PURE={'src/cache_images.py':'8878bfd7c879dd2cbe263496f61f33562b9659bfe1151112b3140b696fb931f5',
 'audit/audit_exact_identity_recovery.py':'4ddc8b7e9ca38c50776295f7b5ba50c1efcdd3cb953df51ab48f584314d68794',
 'audit/build_clean_component_split.py':'27af26630cfe64a54bf0b65976f28dde4754b64bbd73821424830df51fa8cded'}
IDENTITY_URL='https://ftp.mi.fu-berlin.de/pub/cmb-data/celeba/identity_CelebA.txt'
IDENTITY_MD5='32bd1bd63d3c78cd57e08160ec5ed1e2'
IDENTITY_SHA='c6143857c3e2630ac2da9f782e9c1232e5e59be993a9d44e8a7916c78a6158c0'
PRIOR_READY='62347bb0d64d5d8fb816b8c233cdda41be30f89bb92d0a86bc30264f5f3ee639'
COUNTS=[[50146,15293,46234,942],[9944,3074,9885,253],[10545,4127,10372,173]]
CLUSTERS=[[3097,1664,2780,282],[615,327,597,67],[674,377,617,65]]


def require(condition,message):
    if not condition:raise RuntimeError(message)


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path,algorithm='sha256'):
    digest=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def progress(**value):print(json.dumps(value,allow_nan=False),flush=True)


def gate():
    require(ROOT.resolve()==ROOT and not ROOT.is_symlink(),'exact_new_owned_root_required')
    approval_path=ROOT/'data_preparation_approved.json';approval=json.loads(approval_path.read_text())
    require(approval['status']=='approved_before_v3_standalone_canonical_and_clean_rebuild'
        and approval['source_sha256']==sha(__file__) and approval['pure_sources']==PURE
        and approval['expected_input_array_sha256']==EXPECTED and approval['attached_root']==str(ATTACHED)
        and approval['prior_vm_files_absent_observed'] is True and approval['quality_values_seen'] is False,
        'source_bound_reconstruction_approval_required')
    stamp=datetime.fromisoformat(approval['approved_utc'].replace('Z','+00:00'))
    require(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),'approval_timestamp')
    deadline=datetime.fromisoformat(approval['deadline_utc'].replace('Z','+00:00'))
    require(deadline.tzinfo is not None and deadline>datetime.now(timezone.utc),'study_deadline_elapsed')
    for relative,digest in PURE.items():
        path=ROOT/relative;require(not path.is_symlink() and sha(path)==digest,'pure_definition_source_changed')
    return approval,approval_path,deadline


def definitions(relative,name):
    path=ROOT/relative;require(sha(path)==PURE[relative],'pure_source_changed_before_import')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def metadata(np):
    mappings={};records={}
    expected=[f'{i:06d}.jpg' for i in range(1,N+1)]
    for name in ['list_attr_celeba.csv','list_eval_partition.csv']:
        path=ATTACHED/name;require(path.is_file() and 0<path.stat().st_size<=64*1024**2,'bounded_attached_CSV_required')
        raw=path.read_bytes();reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
        fields=reader.fieldnames;needed={'image_id','Male','Blond_Hair'} if 'attr' in name else {'image_id','partition'}
        require(isinstance(fields,list) and len(fields)==len(set(fields)) and needed<=set(fields),'CSV_header_differs')
        values={}
        for row in reader:
            require(len(values)<N and None not in row and all(v is not None for v in row.values()),'bad_CSV_row')
            filename=row['image_id'];require(filename not in values and len(filename)==10 and filename.endswith('.jpg')
                and filename[:6].isdigit() and 1<=int(filename[:6])<=N,'bad_CSV_filename')
            if 'attr' in name:
                require(row['Male'] in ['-1','1'] and row['Blond_Hair'] in ['-1','1'],'bad_attribute_encoding')
                values[filename]=2*int(row['Male']=='1')+int(row['Blond_Hair']=='1')
            else:
                require(row['partition'] in ['0','1','2'],'bad_partition_encoding');values[filename]=int(row['partition'])
        require(set(values)==set(expected),'complete_metadata_filename_join_required')
        mappings[name]=values;records[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'rows':len(values)}
    return (np.asarray(expected),np.asarray([mappings['list_attr_celeba.csv'][n] for n in expected],dtype=np.int64),
        np.asarray([mappings['list_eval_partition.csv'][n] for n in expected],dtype=np.int64),records)


class Redirects(urllib.request.HTTPRedirectHandler):
    def __init__(self):self.count=0
    def redirect_request(self,req,fp,code,msg,headers,url):
        self.count+=1;p=urllib.parse.urlsplit(url)
        require(self.count<=3 and p.scheme=='https' and p.hostname=='ftp.mi.fu-berlin.de'
            and p.port in [None,443] and p.username is None and p.password is None,'identity_redirect_refused')
        return super().redirect_request(req,fp,code,msg,headers,url)


def identity(output):
    # Exclusive reservation precedes the socket; no direct worker retry.
    write(output/'identity_request_reserved.json',{'utc':utc(),'url':IDENTITY_URL,'attempts':1,'auth_cookies_or_confirmation':False})
    redirects=Redirects();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),redirects)
    path=output/'private_identity_CelebA.txt';began=time.monotonic();count=0
    request=urllib.request.Request(IDENTITY_URL,headers={'User-Agent':'ResearchMetadataAudit/1.0'})
    with opener.open(request,timeout=10) as response:
        require(response.status==200 and 'html' not in response.headers.get('Content-Type','').lower(),'identity_HTTP_or_HTML_refused')
        with path.open('xb') as f:
            while True:
                require(time.monotonic()-began<30,'identity_30second_cap')
                block=response.read(65536)
                if not block:break
                count+=len(block);require(count<=8*1024**2,'identity_8MiB_cap');f.write(block)
    require(sha(path,'md5')==IDENTITY_MD5 and sha(path)==IDENTITY_SHA,'official_identity_bytes_mismatch')
    write(output/'identity_acquisition.json',{'status':'authenticated_new_recovery_identity_input','utc':utc(),
        'url':IDENTITY_URL,'bytes':count,'MD5':IDENTITY_MD5,'sha256':IDENTITY_SHA,'redirects':redirects.count,
        'attempts':1,'auth_cookies_proxy_or_confirmation_used':False,'raw_identity_for_export':False})
    return path


def worker():
    approval,approval_path,deadline=gate()
    require((ROOT/'preparation_guard').is_dir(),'owned_coordinator_required')
    write(ROOT/'preparation_guard/worker_reserved.json',{'utc':utc(),'source_sha256':sha(__file__)})
    for path in [ROOT/'data',ROOT/'integrity',SCRATCH]:require(not path.exists() and not path.is_symlink(),'fresh_reconstruction_targets_required')
    (ROOT/'data').mkdir();(ROOT/'integrity').mkdir();SCRATCH.mkdir()
    began=time.monotonic();old=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('reconstruction_inner_cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,min(570,(deadline-datetime.now(timezone.utc)).total_seconds()))
    try:
        import numpy as np
        import PIL
        from PIL import ImageFile
        import scipy
        require(PIL.__version__=='11.3.0' and ImageFile.LOAD_TRUNCATED_IMAGES is False,'fresh_child_requires_Pillow11_3')
        canonical=definitions('src/cache_images.py','_v3_canonical_preprocess')
        exact=definitions('audit/audit_exact_identity_recovery.py','_v3_pure_exact')
        clean=definitions('audit/build_clean_component_split.py','_v3_pure_clean')
        names,groups,splits,csv_records=metadata(np)
        original_index=ROOT/'integrity/index.npz'
        with original_index.open('xb') as f:np.savez(f,filenames=names,groups=groups,splits=splits)
        require(sha(original_index)==EXPECTED['original_index_sha256'],'whole_original_index21dd_mismatch')
        write(ROOT/'integrity/canonical_metadata.json',{'status':'whole_original_index_authenticated','sha256':sha(original_index),
            'CSV_inputs':csv_records,'raw_metadata_for_export':False})
        image_dir=ATTACHED/'img_align_celeba/img_align_celeba'
        require(image_dir.is_dir(),'approved_attached_image_directory_missing')
        temporary=SCRATCH/'images64.incomplete.npy';array=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.uint8,shape=(N,3,64,64))
        try:
            with ThreadPoolExecutor(max_workers=32) as pool:
                for start in range(0,N,1024):
                    paths=[image_dir/f'{i:06d}.jpg' for i in range(start+1,min(start+1024,N)+1)]
                    for offset,pixels in enumerate(pool.map(canonical.preprocess_image,paths)):
                        require(pixels.shape==(3,64,64) and pixels.dtype==np.uint8,'canonical_decoder_shape_type')
                        array[start+offset]=pixels
                    if start%8192==0:progress(phase='canonical_cache',rows=min(start+1024,N),total=N)
            array.flush()
        finally:del array
        require(sha(temporary)==EXPECTED['original_cache_sha256'],'whole_canonical_cache98d39_mismatch')
        cache_path=ROOT/'data/images64.npy';partial=ROOT/'data/images64.incomplete.npy'
        with temporary.open('rb') as source,partial.open('xb') as target:
            for block in iter(lambda:source.read(16*1024**2),b''):target.write(block)
            target.flush();os.fsync(target.fileno())
        require(sha(partial)==EXPECTED['original_cache_sha256'],'streamed_cache_copy_mismatch')
        os.link(partial,cache_path);partial.unlink()
        identity_path=identity(ROOT/'integrity');labels=exact.parse_identities(identity_path,names.tolist())
        require(len(set(labels))==10177,'identity_label_count_mismatch')
        cache=np.load(cache_path,mmap_mode='r',allow_pickle=False)
        pixel_report,classes=exact.scan_pixels(cache,groups,splits,
            lambda n:progress(phase='exact_flip',rows=n,total=N))
        write(ROOT/'integrity/exact_flip_summary.json',pixel_report)
        dsu=clean.DisjointSet(N);dsu.add_classes(labels);dsu.add_classes(classes[0]);dsu.add_classes(classes[1])
        def near_progress(phase,count,total):progress(phase=phase,completed=count,total_or_cap=total)
        hashes=clean.perceptual_hashes(cache,near_progress)
        edges,near_report=clean.near_edges(cache,hashes,dsu,near_progress)
        union=dsu.canonical_ids();new_splits,split_report=clean.frozen_partition(splits,groups,union,classes[1])
        require(split_report['split_group_image_counts']==COUNTS and split_report['split_group_component_counts']==CLUSTERS,
            'reconstructed_split_counts_differ_no_relaxation')
        np.save(ROOT/'data/component_ids.npy',union,allow_pickle=False)
        np.savez_compressed(ROOT/'data/clean_index.npz',filenames=names,groups=groups,splits=new_splits)
        for filename,key in [('component_ids.npy','component_ids_sha256'),('clean_index.npz','clean_index_sha256')]:
            require(sha(ROOT/'data'/filename)==EXPECTED[key],'reconstructed_clean_array_bytes_differ_'+key)
        write(ROOT/'data/near_screen_summary.json',near_report);write(ROOT/'data/split_policy.json',clean.POLICY)
        write(ROOT/'data/split_report.json',split_report)
        for relative,digest in PURE.items():require(sha(ROOT/relative)==digest,'pure_source_changed_during_rebuild')
        require(sha(__file__)==approval['source_sha256'] and sha(cache_path)==EXPECTED['original_cache_sha256']
            and sha(original_index)==EXPECTED['original_index_sha256'],'source_or_canonical_inputs_changed')
        ready={'status':'frozen_clean_component_split','study_id':'generative_recovery_2500','utc':utc(),
            'prior_definition_ready_sha256':PRIOR_READY,'reconstructed_inputs_only':True,'models_recovered':0,
            'old_checkpoints_or_quality_records_claimed_recovered':False,'source_sha256':sha(__file__),
            'pure_sources':PURE,'approval_sha256':sha(approval_path),'preparation_source_path':'runtime/prepare_recovery_data.py',
            **EXPECTED,'cache_path':str(cache_path),'cache_sha256':EXPECTED['original_cache_sha256'],
            'metadata_npz':str(ROOT/'data/clean_index.npz'),'component_ids_path':str(ROOT/'data/component_ids.npy'),
            **split_report,'original_val_test_images_used':False,'test_quality_accessed':False,'quality_results_accessed':False,
            'declared_near_relation_completed':True,'inputs_unchanged':True,
            'near_screen_summary_sha256':sha(ROOT/'data/near_screen_summary.json'),
            'split_report_sha256':sha(ROOT/'data/split_report.json'),'split_policy_sha256':sha(ROOT/'data/split_policy.json'),
            'identity_acquisition_sha256':sha(ROOT/'integrity/identity_acquisition.json'),
            'exact_flip_summary_sha256':sha(ROOT/'integrity/exact_flip_summary.json'),
            'identity_label_audit_status':'official_MD5_and_SHA256_complete_filename_join',
            'environment':{'numpy':np.__version__,'Pillow':PIL.__version__,'scipy':scipy.__version__},
            'elapsed_seconds':time.monotonic()-began,'Torch_or_models_imported':False,
            'relation_scope':'supplied_identity_exact_flip_declared_phash4_RGB_RMSE8_not_universal_semantic_similarity',
            'raw_face_images_or_full_identity_metadata_for_export':False}
        write(ROOT/'data/data_ready.json',ready);progress(status='frozen_clean_component_split',ready_sha256=sha(ROOT/'data/data_ready.json'))
    except BaseException as error:
        # Avoid exception messages carrying download URLs or raw row values.
        write(ROOT/'integrity/failure.json',{'status':'INCOMPLETE_NO_TRAINING_ALLOWED','utc':utc(),
            'error_type':type(error).__name__,'no_retry':True,'partial_inputs_preserved':True,'quality_values_seen':False})
        raise RuntimeError('V3_DATA_PREPARATION_FAILED_'+type(error).__name__) from None
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def coordinator():
    approval,approval_path,deadline=gate();guard=ROOT/'preparation_guard'
    require(not guard.exists() and not guard.is_symlink(),'fresh_guard_required')
    guard.mkdir();began=time.monotonic();code=None;timed_out=False;process=None
    write(guard/'started.json',{'utc':utc(),'source_sha256':sha(__file__),'approval_sha256':sha(approval_path),'cap_seconds':600})
    try:
        with (guard/'stdout.log').open('xb') as stdout,(guard/'stderr.log').open('xb') as stderr:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
                stdout=stdout,stderr=stderr,start_new_session=True)
            try:code=process.wait(timeout=min(600,max(1,(deadline-datetime.now(timezone.utc)).total_seconds())))
            except subprocess.TimeoutExpired:timed_out=True
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                code=process.returncode
    finally:
        ready=ROOT/'data/data_ready.json'
        result={'status':'COMPLETED' if code==0 and not timed_out and ready.is_file() else 'INCOMPLETE',
            'return_code':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-began,
            'source_sha256':sha(__file__),'ready_sha256':sha(ready) if ready.is_file() else None,'no_automatic_retry':True}
        write(guard/'process.json',result)
    print(json.dumps(result),flush=True)
    if result['status']!='COMPLETED':raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    require(sys.flags.optimize==0,'optimized_mode_refused')
    if args.worker:worker()
    else:coordinator()
