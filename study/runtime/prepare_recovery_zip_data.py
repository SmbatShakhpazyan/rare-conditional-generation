"""NEW standalone recovery of canonical inputs and the SAME clean split.
NEW local public-ZIP recovery after the observed incomplete attached-FUSE attempt.
Preserves all first-attempt files; one approved ZIP GET and one identity GET.
No old-run gates, models, Torch, GPU, quality reads, retries or threshold changes.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import csv
import http.cookiejar
import stat
import zipfile
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
SCRATCH=Path('/tmp/generative_recovery_2500_zip')
FAILED_SOURCE='1bfc098f30aeb3c5ee155f4d056dac82f0f0e3e4c2b11c26f1bee384791daa97'
ZIP_URL='https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset'
ZIP_SHA='4ceccf7dffb6f3b40559fb9ae68b42a71c3528b2881b787d65f4d1f75329f02d'
ZIP_BYTES=1427750792
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
    approval_path=ROOT/'zip_data_preparation_approved.json';approval=json.loads(approval_path.read_text())
    require(approval['status']=='approved_before_v3_local_ZIP_recovery_after_observed_FUSE_failure'
        and approval['source_sha256']==sha(__file__) and approval['pure_sources']==PURE
        and approval['expected_input_array_sha256']==EXPECTED
        and approval['failed_source_sha256']==FAILED_SOURCE and approval['prior_worker_stopped_confirmed'] is True
        and approval['public_archive_sha256']==ZIP_SHA and approval['public_archive_bytes']==ZIP_BYTES
        and approval['quality_values_seen'] is False,
        'source_bound_reconstruction_approval_required')
    old_process=ROOT/'preparation_guard/process.json'
    require(sha(ROOT/'runtime/prepare_recovery_data.py')==FAILED_SOURCE
        and sha(old_process)==approval['failed_process_record_sha256'],'preserved_failed_attempt_binding')
    record=json.loads(old_process.read_text())
    require(record['status']=='INCOMPLETE' and record['return_code']!=0,'actual_incomplete_prior_attempt_required')
    observed=ROOT/'runtime/fuse_preparation_failure_observed.json'
    require(sha(observed)==approval['observed_failure_sha256'],'actual_observed_failure_required')
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
    # Reuse only the whole authenticated canonical index; no attached FUSE image reads.
    old=ROOT/'integrity_failed_fuse/index.npz'
    require(sha(old)==EXPECTED['original_index_sha256'],'failed_attempt_index21dd_required')
    with np.load(old,allow_pickle=False) as record:
        names=record['filenames'].copy();groups=record['groups'].copy();splits=record['splits'].copy()
    require(names.tolist()==[f'{i:06d}.jpg' for i in range(1,N+1)]
        and groups.shape==(N,) and splits.shape==(N,),'original_metadata_schema')
    return names,groups,splits,{'authenticated_original_index':{'sha256':sha(old),'rows':N,
        'source':'preserved_incomplete_FUSE_attempt','CSV_inputs_reread':False}}


class NoCookies(http.cookiejar.DefaultCookiePolicy):
    def set_ok(self,*args,**kwargs):return False
    def return_ok(self,*args,**kwargs):return False


def safe_archive_url(url):
    parts=urllib.parse.urlsplit(url)
    require(parts.scheme=='https' and parts.hostname in {'www.kaggle.com','kaggle.com','storage.googleapis.com'}
        and parts.port in [None,443] and parts.username is None and parts.password is None
        and not parts.fragment,'public_archive_redirect_refused')
    return parts.hostname


def download_archive(output):
    import requests
    write(output/'archive_request_reserved.json',{'utc':utc(),'endpoint_without_query':ZIP_URL,'attempts':1})
    path=SCRATCH/'public_dataset.zip';partial=SCRATCH/'public_dataset.incomplete.zip'
    began=time.monotonic();count=0;hosts=[];statuses=[];url=ZIP_URL
    with requests.Session() as session:
        session.trust_env=False;session.auth=None;session.cookies.clear();session.cookies.set_policy(NoCookies())
        session.headers.update({'Accept':'application/zip,application/octet-stream','Accept-Encoding':'identity',
            'User-Agent':'source-bound-public-dataset-recovery/1'})
        session.mount('https://',requests.adapters.HTTPAdapter(max_retries=0))
        for hop in range(6):
            hosts.append(safe_archive_url(url));session.cookies.clear()
            remaining=180-(time.monotonic()-began);require(remaining>0,'public_archive_180second_cap')
            with session.get(url,stream=True,allow_redirects=False,timeout=(10,min(20,remaining))) as response:
                statuses.append(response.status_code)
                if response.status_code in [301,302,303,307,308]:
                    require(hop<5 and response.headers.get('Location'),'public_archive_redirect_cap')
                    url=urllib.parse.urljoin(url,response.headers['Location']);safe_archive_url(url);continue
                require(response.status_code==200,'public_archive_HTTP_refused')
                require(response.headers.get('Content-Type','').split(';')[0].lower() in
                    ['application/zip','application/x-zip-compressed','application/octet-stream']
                    and response.headers.get('Content-Encoding','identity').lower()=='identity','archive_content_refused')
                declared=response.headers.get('Content-Length')
                require(declared is None or declared==str(ZIP_BYTES),'public_archive_declared_size_differs')
                digest=hashlib.sha256()
                with partial.open('xb') as stream:
                    for block in response.iter_content(chunk_size=1024**2):
                        if not block:continue
                        count+=len(block);require(count<=ZIP_BYTES and time.monotonic()-began<180,'archive_stream_cap')
                        digest.update(block);stream.write(block)
                        if count//(128*1024**2)!=(count-len(block))//(128*1024**2):
                            progress(phase='local_ZIP_download',bytes=count,total=ZIP_BYTES)
                    stream.flush();os.fsync(stream.fileno())
                require(count==ZIP_BYTES and digest.hexdigest()==ZIP_SHA,'whole_public_archive_identity_mismatch')
                os.link(partial,path);partial.unlink()
                record={'status':'authenticated_public_Kaggle_ZIP','utc':utc(),'sha256':ZIP_SHA,'bytes':count,
                    'endpoint_without_query':ZIP_URL,'redirect_hosts_only':hosts,'HTTP_statuses':statuses,
                    'attempts':1,'retries':0,'auth_cookies_proxy_or_netrc_used':False,
                    'original_author_ZIP_claimed':False,'elapsed_seconds':time.monotonic()-began}
                write(output/'archive_download.json',record);return path
    raise RuntimeError('public_archive_no_final_response')


def selected_members(archive):
    infos=archive.infolist();require(len(infos)<=N+256,'bounded_ZIP_members')
    selected={};seen=set();prefixes=set();expanded=0
    for item in infos:
        name=item.filename;parts=Path(name.rstrip('/')).parts
        require(name and name not in seen and '\\' not in name and ':' not in name and '\x00' not in name
            and not name.startswith('/') and '..' not in parts,'unsafe_or_duplicate_ZIP_member')
        seen.add(name);mode=item.external_attr>>16
        require(not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in [0,stat.S_IFREG,stat.S_IFDIR]
            and not item.flag_bits&1 and item.compress_type in [zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED],
            'ZIP_special_encrypted_or_compression_refused')
        expanded+=item.file_size;require(0<=item.file_size<=64*1024**2 and expanded<=8*1024**3,'ZIP_expansion_cap')
        if item.is_dir():continue
        if name.lower().endswith(('.jpg','.jpeg')):
            filename=parts[-1];prefix=name[:-len(filename)]
            require(prefix in ['img_align_celeba/','img_align_celeba/img_align_celeba/']
                and len(filename)==10 and filename[:6].isdigit() and filename.endswith('.jpg')
                and 1<=int(filename[:6])<=N and filename not in selected and 0<item.file_size<=4*1024**2,
                'unexpected_JPEG_member')
            selected[filename]=item;prefixes.add(prefix)
    require(set(selected)=={f'{i:06d}.jpg' for i in range(1,N+1)} and len(prefixes)==1,'all_exact202599_JPEGs_required')
    return selected,{'archive_members':len(infos),'selected_JPEGs':N,'selected_prefix':next(iter(prefixes)),
        'declared_expanded_bytes':expanded,'filesystem_extraction_used':False,'FUSE_JPEG_reads':0}


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
    require((ROOT/'zip_preparation_guard').is_dir(),'owned_coordinator_required')
    write(ROOT/'zip_preparation_guard/worker_reserved.json',{'utc':utc(),'source_sha256':sha(__file__)})
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
        archive_path=download_archive(ROOT/'integrity')
        temporary=SCRATCH/'images64.incomplete.npy';array=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.uint8,shape=(N,3,64,64))
        try:
            # ZipFile serializes the short seek/read; Pillow JPEG decode runs in the pool on local bytes.
            with zipfile.ZipFile(archive_path) as archive:
                members,selection=selected_members(archive)
                write(ROOT/'integrity/archive_selection.json',selection)
                def decode(filename):
                    raw=archive.read(members[filename])
                    return canonical.preprocess_image(io.BytesIO(raw))
                with ThreadPoolExecutor(max_workers=16) as pool:
                    for start in range(0,N,1024):
                        filenames=[f'{i:06d}.jpg' for i in range(start+1,min(start+1024,N)+1)]
                        for offset,pixels in enumerate(pool.map(decode,filenames)):
                            require(pixels.shape==(3,64,64) and pixels.dtype==np.uint8,'canonical_decoder_shape_type')
                            array[start+offset]=pixels
                        if start%8192==0:progress(phase='local_ZIP_canonical_cache',rows=min(start+1024,N),total=N)
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
            'pure_sources':PURE,'approval_sha256':sha(approval_path),'preparation_source_path':'runtime/prepare_recovery_zip_data.py',
            'prior_failed_FUSE_source_sha256':FAILED_SOURCE,'prior_failed_process_sha256':approval['failed_process_record_sha256'],
            'recovery_method':'one_authenticated_public_ZIP_local_bytes_no_FUSE_JPEG_reads',
            'archive_sha256':ZIP_SHA,'archive_bytes':ZIP_BYTES,
            'archive_download_sha256':sha(ROOT/'integrity/archive_download.json'),
            'archive_selection_sha256':sha(ROOT/'integrity/archive_selection.json'),
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
    approval,approval_path,deadline=gate();guard=ROOT/'zip_preparation_guard'
    require(not guard.exists() and not guard.is_symlink(),'fresh_guard_required')
    require(not (ROOT/'data/data_ready.json').exists(),'prior_attempt_must_not_have_a_ready_record')
    for name in ['data','integrity']:
        old=ROOT/name;preserved=ROOT/(name+'_failed_fuse')
        require(old.is_dir() and not old.is_symlink() and not preserved.exists() and not preserved.is_symlink(),
            'fresh_named_failed_attempt_preservation_required')
    require(sha(ROOT/'integrity/index.npz')==EXPECTED['original_index_sha256'],'prior_index21dd_required')
    guard.mkdir()
    snapshot={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'preparation_guard/process.json',ROOT/'integrity/index.npz']}
    for name in ['data','integrity']:(ROOT/name).rename(ROOT/(name+'_failed_fuse'))
    write(guard/'preserved_failed_attempt.json',{'utc':utc(),'source_sha256':FAILED_SOURCE,'prior_files':snapshot,
        'directories_renamed_without_deletion':{'data':'data_failed_fuse','integrity':'integrity_failed_fuse'},
        'prior_scratch_unchanged':'/tmp/generative_recovery_2500'})
    began=time.monotonic();code=None;timed_out=False;process=None
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
