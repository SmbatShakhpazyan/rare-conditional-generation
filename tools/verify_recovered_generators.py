"""Forward-only CPU verification of the four already trained EMA generators.

This is an artifact check, not a new benchmark or a feature-bank replay.
Never deserializes arbitrary objects: torch.load uses weights_only=True.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import time
import torch

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--weights-dir',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('archived_models',args.weights_dir/'models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reports=[];start=time.perf_counter()
    for p in sorted(args.weights_dir.glob('*_EMA.pt')):
        payload=torch.load(p,map_location='cpu',weights_only=True)
        if not isinstance(payload,dict):raise TypeError('Expected an archived state dictionary')
        # Archives wrap the inference state_dict with immutable producer metadata.
        config=payload['generator_config']
        assert payload['study_id']=='generative_recovery_1500' and payload['step']==1500
        assert config['z_dim']==128 and config['num_classes']==4 and config['base_channels']==32
        g=module.Generator(**{k:config[k] for k in ['z_dim','num_classes','base_channels']}).eval()
        g.load_state_dict(payload['ema_generator'],strict=True)
        assert module.parameter_count(g)==737315
        before={k:v.detach().clone() for k,v in g.state_dict().items()}
        torch.manual_seed(20261119);z=torch.randn(8,128)
        labels=torch.tensor([0,1,2,3,0,1,2,3],dtype=torch.long)
        with torch.no_grad():
            image=g(z,labels);again=g(z,labels)
        assert image.shape==(8,3,64,64)
        assert torch.isfinite(image).all() and image.min()>=-1 and image.max()<=1
        assert torch.equal(image,again)
        assert all(torch.equal(v,g.state_dict()[k]) for k,v in before.items())
        reports.append(dict(file=p.name,sha256=digest(p),bytes=p.stat().st_size,
                            strict_state_load=True,parameters=module.parameter_count(g),
                            output_shape=list(image.shape),finite=True,
                            output_min=float(image.min()),output_max=float(image.max()),
                            same_runtime_repeat_equal=True,state_unchanged=True,
                            output_float32_sha256=hashlib.sha256(image.numpy().tobytes()).hexdigest()))
    assert len(reports)==4
    result=dict(status='passed',type='forward_only_artifact_verification',
                python=platform.python_version(),torch=torch.__version__,device='cpu',
                source_models_sha256=digest(args.weights_dir/'models.py'),
                latent_seed=20261119,images_per_generator=8,models=reports,
                elapsed_seconds=time.perf_counter()-start,training_updates=0,
                efficacy_metrics_recomputed=False,full_KID_CI_replay=False,
                missing_models=['B_seed202','B_seed303'],
                limitations=['No independent KID PRDC or bootstrap replay',
                             'No cross-device or cross-version byte identity guarantee',
                             'No semantic diversity or privacy evaluation'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'generators':len(reports),
                      'training_updates':0,'elapsed_seconds':result['elapsed_seconds']}))

if __name__=='__main__':main()
