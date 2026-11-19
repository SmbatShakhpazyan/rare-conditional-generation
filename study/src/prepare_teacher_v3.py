"""Freeze the predeclared B-development D and one full TRAIN-only moment pass.

No work on import. Never evaluates validation/test or chooses a teacher by score.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/kaggle/working/generative_recovery_2500')

def require(value,message):
    if not value:raise RuntimeError(message)

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):result.update(block)
    return result.hexdigest()

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    require(not root.is_symlink() and root.resolve()==ROOT,'Exact v3 root required')
    plan=json.loads((root/'study_plan_frozen.json').read_text())
    require(plan['study_id']=='generative_recovery_1500' and plan['quality_results_seen'] is False,
        'Prospective study plan required')
    require(sha(__file__)==plan['auxiliary_sources']['src/prepare_teacher_v3.py'],'Teacher preparation source differs')
    for name,digest in plan['training_sources'].items():
        require(sha(root/'src'/name)==digest,'Training source changed: '+name)
    folder=root/'development/B_seed909';complete_path=folder/'complete.json'
    record=json.loads(complete_path.read_text());config=record['config']
    require(record['status']=='complete' and record['arm']=='B' and record['seed']==909
        and record['completed_steps']==1500 and config['phase']=='development'
        and config['coefficient']==0. and config['source_sha256']==plan['training_sources'],
        'Only fixed last B-development seed9091500 can supply teacher')
    require(sha(folder/'latest.pt')==record['checkpoint_sha256']
        and sha(folder/'D_final.pt')==record['D_final_sha256'],'Teacher source artifacts changed')
    for key,hash_key in [('metadata_npz','index_sha256'),('component_ids_path','component_ids_sha256'),
                         ('cache_path','cache_sha256')]:
        require(sha(config[key])==config[hash_key],'Bound teacher data changed')
    output=root/'teacher';require(not output.exists() and not output.is_symlink(),'Preserve previous teacher attempt')
    output.mkdir();started=time.monotonic()
    import numpy as np
    import torch
    from models import ProjectionDiscriminator
    from interaction_statistics import compute_real_statistics
    torch.set_num_threads(2)
    require(torch.cuda.is_available(),'Actual GPU required')
    artifact=torch.load(folder/'D_final.pt',map_location='cpu',weights_only=True)
    require(artifact['arm']=='B' and artifact['seed']==909 and artifact['step']==1500
        and artifact['source_sha256']==plan['training_sources'],'Actual D-only export differs')
    target=output/'fixed_teacher.pt'
    # Exact tensor state including all SN originals/u/v; no power iteration/EMA.
    torch.save(artifact,target)
    teacher=ProjectionDiscriminator(num_classes=4,base_channels=32).cuda()
    teacher.load_state_dict(artifact['discriminator'],strict=True)
    teacher.eval().requires_grad_(False)
    before={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
    with np.load(config['metadata_npz'],allow_pickle=False) as index:
        groups=index['groups'];splits=index['splits']
    components=np.load(config['component_ids_path'],allow_pickle=False)
    cache=np.load(config['cache_path'],allow_pickle=False,mmap_mode='r')
    ids=np.flatnonzero(splits==0)
    require(components.shape==(202599,) and components.dtype==np.int64,'Wrong union components')
    features=np.empty((len(ids),128),dtype=np.float32)
    with torch.no_grad():
        for start in range(0,len(ids),64):
            chosen=ids[start:start+64]
            x=torch.from_numpy(np.array(cache[chosen],copy=True)).cuda().float().div_(127.5).sub_(1)
            y=torch.from_numpy(groups[chosen].astype(np.int64)).cuda()
            _,h=teacher(x,y,return_features=True)
            require(h.shape==(len(chosen),128) and bool(torch.isfinite(h).all()),'Teacher feature shape/finite failure')
            features[start:start+len(chosen)]=h.cpu().numpy()
            if start%8192==0:print(json.dumps({'phase':'train_only_teacher_features','images':min(start+64,len(ids)),
                'total':len(ids)}),flush=True)
    require(all(torch.equal(v.cpu(),before[k]) for k,v in teacher.state_dict().items()),
        'Frozen teacher state changed during reference extraction')
    statistics=compute_real_statistics(features,groups[ids],components[ids],scale_floor=.001)
    result={'status':'fixed_train_only_teacher_statistics','study_id':'generative_recovery_1500',
        'teacher_selection':'last_raw_discriminator_B_seed909_1500_no_validation_selection',
        'teacher_artifact_sha256':sha(target),'teacher_complete_sha256':sha(complete_path),
        'teacher_source_checkpoint_sha256':record['checkpoint_sha256'],
        'teacher_D_export_sha256':record['D_final_sha256'],
        'index_sha256':config['index_sha256'],'component_ids_sha256':config['component_ids_sha256'],
        'cache_sha256':config['cache_sha256'],'statistics':statistics,
        'source_sha256':plan['training_sources'],'preparation_source_sha256':sha(__file__),
        'reference_orientation':'canonical_unaugmented_RGB64','train_images_once':len(ids),
        'validation_or_test_images_used':0,'teacher_state_unchanged':True,
        'confidence_bound':False,'elapsed_seconds':time.monotonic()-started,
        'numpy':np.__version__,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
    write(output/'statistics.json',result)
    write(output/'complete.json',{'status':'complete','teacher_artifact_sha256':sha(target),
        'statistics_sha256':sha(output/'statistics.json'),'source_plan_sha256':sha(root/'study_plan_frozen.json')})
    print(json.dumps({'status':'fixed_teacher_complete','statistics_sha256':sha(output/'statistics.json')}),flush=True)

if __name__=='__main__':main()
