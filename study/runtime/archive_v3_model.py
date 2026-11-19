"""Stdlib-only immediate selected-EMA archive of one completed v2 model.

No Torch, network, dataset pixels, model, feature or quality calls. A fresh
durable directory is mandatory; failures preserve partials without auto retry.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import signal
import tempfile
import zipfile

ROOT = Path('/kaggle/working/generative_recovery_2500')
CHUNK_LIMIT = 512 * 1024
RAW_CHUNK_BYTES = (CHUNK_LIMIT // 77) * 57
STUDY = 'generative_recovery_1500'
NOTICE = '''PRIVATE RESEARCH ARTIFACT; NO BLANKET PUBLIC LICENSE GRANT

This archive contains one newly trained, selected EMA generator and the
corresponding source and metadata. It contains no raw CelebA images, raw
annotations, component/identity IDs, pretrained teacher/feature weights,
discriminator, optimizer or browser/authentication state.

CelebA https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html is restricted to
non-commercial research with redistribution restrictions. This private
delivery grants no new dataset, image, identity, publicity, model-weight or
commercial-use rights. Trained weights/generated images are not certified
free of memorization or privacy risk. Conditions are literal annotation bins.
PyTorch and other dependencies retain their own licenses. Standard residual,
projection/SN/cBN, feature matching, MMD, path normalization and gSR antecedents
remain credited in the accompanying research. No novelty, fairness, semantic
correctness, universal superiority or privacy certificate is granted.
'''

INFERENCE = '''"""Selected EMA inference only; no training data/teacher/feature model."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import Generator
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True)
p.add_argument('--device',default='cpu')
p.add_argument('--seed',type=int,default=78231)
p.add_argument('--per-group',type=int,default=16)
a=p.parse_args()
if a.per_group<1:raise ValueError('per-group must be positive')
record=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
cfg=record['generator_config']
if cfg.get('image_size',64)!=64:raise ValueError('RGB64 generator required')
g=Generator(**{k:cfg[k] for k in ('z_dim','num_classes','base_channels')})
g.load_state_dict(record['ema_generator'],strict=True)
g.eval().requires_grad_(False).to(a.device)
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
for group in range(4):
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([a.seed,group,80317])))
    z=rng.standard_normal((a.per_group,cfg['z_dim']),dtype=np.float32)
    with torch.inference_mode():
        images=g(torch.from_numpy(z).to(a.device),torch.full((a.per_group,),group,dtype=torch.long,device=a.device))
    images=images.float().cpu().permute(0,2,3,1).numpy()
    if not np.isfinite(images).all() or images.min() < -1 or images.max() > 1:
        raise ValueError('Invalid generator pixels')
    pixels=np.rint((images.astype(np.float64)+1)*127.5).astype(np.uint8)
    for i,image in enumerate(pixels):Image.fromarray(image).save(out/f'group{group}_{i:04d}.png')
print('Generated all requested bins without filtering; no semantic-label certificate.')
'''


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def file_record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/symlink input')
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def validate_run(root, key):
    root = Path(root)
    require(not root.is_symlink() and root.resolve() == ROOT, 'Exact v2 root required')
    require(re.fullmatch(r'(B|C|C0_matched)_seed(101|202|303)', key), 'Unexpected confirmation run')
    plan_path = root/'study_plan_frozen.json'
    plan = read(plan_path)
    require(plan['status'] == 'frozen_before_teacher_grid_validation_test'
            and plan['study_id'] == STUDY, 'Wrong study plan')
    require(plan['auxiliary_sources']['runtime/archive_v3_model.py'] == sha(__file__), 'Unapproved archiver')
    inputs = {str(plan_path): sha(plan_path)}
    for name, digest in plan['training_sources'].items():
        require(Path(name).name == name, 'Unsafe source name')
        path = root/'src'/name
        require(file_record(path)['sha256'] == digest, 'Training source changed')
        inputs[str(path)] = digest
    selection_path = root/'selection_frozen.json'
    selection = read(selection_path)
    require(selection['status'] == 'frozen_validation_only_selection_before_confirmation_and_test'
            and selection['study_plan_sha256'] == sha(plan_path)
            and selection['test_quality_seen'] is False, 'Selection not locked')
    inputs[str(selection_path)] = sha(selection_path)
    confirmation_path = root/'confirmation_plan_frozen.json'
    confirmation = read(confirmation_path)
    require(confirmation['status'] == 'frozen_before_confirmation_and_test'
            and confirmation['selection_sha256'] == sha(selection_path), 'Confirmation plan differs')
    inputs[str(confirmation_path)] = sha(confirmation_path)
    matches = [c for c in confirmation['jobs'] if c['run_id'] == key]
    require(len(matches) == 1, 'Exactly one frozen confirmation job required')
    config = matches[0]
    require(config['phase'] == 'confirmation' and config['steps'] in (1500,)
            and config['seed'] in (101,202,303) and config['source_sha256'] == plan['training_sources']
            and config['selection_sha256'] == sha(selection_path), 'Wrong frozen job')
    method = key.rsplit('_seed',1)[0]
    selected = selection['selected']['C' if method == 'C0_matched' else method]
    require(config['coefficient'] == selected['coefficient']
            and config['selected_ema_decay'] == selected['ema_decay']
            and config['kappa'] == (0. if method == 'C0_matched' else selected['kappa']), 'Selected settings differ')
    if method == 'C0_matched':
        require(selected['kappa'] == 1. and config['arm'] == 'C', 'Redundant control not authorized')
    else:
        require(config['arm'] == method, 'Arm differs')
    ready_path = root/'data/data_ready.json'
    ready = read(ready_path)
    require(file_record(ready_path)['sha256'] == config['data_ready_sha256'] == plan['data_ready_sha256']
            and ready['status'] == 'frozen_clean_component_split'
            and ready['test_quality_accessed'] is False, 'Clean data binding changed')
    inputs[str(ready_path)] = sha(ready_path)
    for name, digest_key in [('clean_index.npz','clean_index_sha256'), ('component_ids.npy','component_ids_sha256')]:
        path = root/'data'/name
        require(file_record(path)['sha256'] == ready[digest_key], 'Private split bytes changed')
        inputs[str(path)] = ready[digest_key]
    require(config['index_sha256'] == ready['clean_index_sha256']
            and config['component_ids_sha256'] == ready['component_ids_sha256'], 'Config data identity differs')
    folder = root/'confirmation'/key
    require(not folder.is_symlink(), 'Run alias refused')
    complete_path = folder/'complete.json'
    complete = read(complete_path)
    require(complete['status'] == 'complete' and complete['config'] == config
            and complete['completed_steps'] == config['steps'] and complete['arm'] == config['arm']
            and complete['seed'] == config['seed'] and complete['pilot'] is False
            and complete['generator_parameters'] == 737315, 'Actual completion differs')
    inputs[str(complete_path)] = file_record(complete_path)['sha256']
    require(file_record(folder/'latest.pt')['sha256'] == complete['checkpoint_sha256'], 'Source checkpoint changed')
    inputs[str(folder/'latest.pt')] = complete['checkpoint_sha256']
    decay = str(config['selected_ema_decay'])
    require(decay in ('0.99','0.999'), 'EMA outside fixed grid')
    ema = folder/('EMA_'+decay+'.pt')
    item = complete['ema_artifacts'][decay]
    require(Path(item['path']) == ema and file_record(ema)['sha256'] == item['sha256'], 'EMA bytes differ')
    inputs[str(ema)] = item['sha256']
    return config, complete, ema, inputs


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate members')
        manifest = json.loads(archive.read('MANIFEST.json'))
        require(set(names) == set(manifest['files']) | {'MANIFEST.json'}, 'ZIP membership differs')
        for name, record in manifest['files'].items():
            require(Path(name).name == name and name not in ('.','..') and '\\' not in name, 'Unsafe member')
            data = archive.read(name)
            require(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], 'ZIP payload differs')
    return manifest


def make_archive(destination, files, metadata):
    require(not destination.exists() and not destination.is_symlink(), 'Refuse ZIP overwrite')
    manifest = {'metadata': metadata, 'manifest_excludes_itself': True,
                'files': {name:file_record(path) for name,path in sorted(files.items())}}
    require('MANIFEST.json' not in files and all(Path(n).name == n for n in files), 'Flat payload allowlist required')
    partial = destination.with_name(destination.name+'.partial')
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name,path in sorted(files.items()):
            info = zipfile.ZipInfo(name,(2026,11,19,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            with Path(path).open('rb') as source, archive.open(info,'w') as target:
                shutil.copyfileobj(source,target,1024*1024)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n')
    verify_archive(partial)
    partial.replace(destination)
    return manifest


def armour(archive, folder):
    original = file_record(archive)
    prefix = archive.stem+'--'+original['sha256'][:16]
    output = folder/(prefix+'.b64chunks'); output.mkdir(exist_ok=False)
    records=[]; decoded=hashlib.sha256(); byte_count=0
    with archive.open('rb') as source:
        for number, raw in enumerate(iter(lambda:source.read(RAW_CHUNK_BYTES),b''),1):
            encoded=base64.b64encode(raw)
            text=b'\n'.join(encoded[i:i+76] for i in range(0,len(encoded),76))+b'\n'
            require(text.isascii() and len(text)<=CHUNK_LIMIT, 'ASCII cap failed')
            name=f'{prefix}--chunk{number:05d}.b64.txt'
            with (output/name).open('xb') as target: target.write(text)
            copied=(output/name).read_bytes()
            restored=base64.b64decode(copied.replace(b'\n',b''),validate=True)
            require(restored==raw, 'Chunk roundtrip differs')
            decoded.update(restored); byte_count+=len(restored)
            records.append({'filename':name,**file_record(output/name),'decoded_bytes':len(raw),
                            'decoded_sha256':hashlib.sha256(raw).hexdigest()})
    require(decoded.hexdigest()==original['sha256'] and byte_count==original['bytes']
            and file_record(archive)==original, 'Whole decoded archive differs')
    manifest={'archive_name':archive.name,'archive_sha256':original['sha256'],
        'archive_bytes':original['bytes'],'chunk_count':len(records),
        'max_encoded_bytes_per_chunk':CHUNK_LIMIT,'columns':76,
        'encoding':'base64_ASCII_LF_final_newline','chunks':records}
    write(output/'CHUNKS.json',manifest)
    return output/'CHUNKS.json'


def archive_model(root,key):
    config, complete, ema, inputs = validate_run(root,key)
    root=Path(root)
    folder=root/'durable'/key
    require(not folder.exists() and not folder.is_symlink(), 'Durable attempt already exists; preserve it')
    folder.mkdir(parents=True); stage=folder/'payload'; stage.mkdir()
    try:
        target=stage/(key+'_EMA.pt'); shutil.copyfile(ema,target)
        require(sha(target)==inputs[str(ema)], 'Copied selected EMA differs')
        shutil.copyfile(root/'src/models.py',stage/'models.py')
        (stage/'generate.py').write_text(INFERENCE)
        (stage/'LICENSE_NOTICE.txt').write_text(NOTICE)
        write(stage/'config.json',config)
        shutil.copyfile(root/'confirmation'/key/'complete.json',stage/'training_complete.json')
        write(stage/'MODEL_RECORD.json',{'study_id':STUDY,'run':key,'arm':config['arm'],'seed':config['seed'],
            'step':config['steps'],'step_definition':'attempted_D_G_iterations',
            'selected_ema_decay':config['selected_ema_decay'],'EMA_file':target.name,'EMA_sha256':sha(target),
            'EMA_bytes':target.stat().st_size,'original_full_checkpoint_sha256':complete['checkpoint_sha256'],
            'complete_record_sha256':inputs[str(root/'confirmation'/key/'complete.json')],
            'study_plan_sha256':inputs[str(root/'study_plan_frozen.json')],
            'selection_sha256':config['selection_sha256'],'data_ready_sha256':config['data_ready_sha256'],
            'clean_index_sha256':config['index_sha256'],'component_ids_sha256':config['component_ids_sha256'],
            'models_py_sha256':config['source_sha256']['models.py'],
            'EMA_bytes_copied_unchanged':True,'Torch_payload_loaded_by_archiver':False,
            'training_producer': 'source_bound_train_v2_minimal_CPU_EMA_artifact'})
        files={p.name:p for p in stage.iterdir()}
        destination=folder/(key+'_EMA.zip')
        make_archive(destination,files,{'study_id':STUDY,'run':key,'step':config['steps'],
            'selected_ema_decay':config['selected_ema_decay'],'no_quality_metrics_used':True})
        chunks=armour(destination,folder)
        require(all(sha(path)==digest for path,digest in inputs.items()), 'Inputs changed during archive')
        result={'status':'durable_selected_EMA_archive_and_text_chunks_verified','study_id':STUDY,'run':key,
            'archive':destination.name,'archive_sha256':sha(destination),'archive_bytes':destination.stat().st_size,
            'armour_manifest':chunks.relative_to(folder).as_posix(),'armour_manifest_sha256':sha(chunks),
            'ASCII_chunk_max_bytes':CHUNK_LIMIT,'selected_ema_decay':config['selected_ema_decay'],
            'EMA_sha256':sha(target),'original_full_checkpoint_sha256':complete['checkpoint_sha256'],
            'no_raw_data_or_pretrained_weights':True,'source_sha256':sha(__file__)}
        write(folder/'complete.json',result)
        print(json.dumps(result),flush=True)
        return result
    except BaseException as error:
        write(folder/'failure.json',{'status':'INCOMPLETE_preserved_no_automatic_retry','error_type':type(error).__name__})
        raise


def self_test():
    import ast
    ast.parse(INFERENCE)
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary); source=root/'opaque.bin'
        source.write_bytes(b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(40000)))
        archive=root/'toy.zip'; make_archive(archive,{'opaque.bin':source},{'toy':True})
        chunks=armour(archive,root); manifest=read(chunks)
        require(manifest['chunk_count']>1, 'Multichunk fixture required')
        rebuilt=b''.join(base64.b64decode((chunks.parent/c['filename']).read_bytes().replace(b'\n',b''),validate=True)
                         for c in manifest['chunks'])
        require(hashlib.sha256(rebuilt).hexdigest()==sha(archive), 'Toy whole roundtrip failed')
        with zipfile.ZipFile(archive,'a') as target:target.writestr('extra.txt','invalid')
        try:verify_archive(archive)
        except RuntimeError:pass
        else:raise RuntimeError('Extra-member tamper accepted')
    print('PASS stdlib multichunk/ZIP/payload/tamper/inference-syntax fixtures; no Torch/data/models')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--run')
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    else:
        require(args.run is not None,'--run required')
        def timeout(*_):raise TimeoutError('55second_archiver_cap')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(55)
        try:archive_model(args.root,args.run)
        finally:signal.alarm(0)
