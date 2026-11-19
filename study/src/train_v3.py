"""V3 recovery clean-split runner with common paired-latent G batches.

The throughput pilot is separate from comparative runs and emits no quality
metrics or sample grids. Main runs require a frozen protocol JSON.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from models import Generator, ProjectionDiscriminator
from model_losses import (hinge_discriminator_loss, generator_adversarial_loss,
                          coarse_interaction_penalty)
from compensated_interaction import compensated_interaction_penalty


def json_dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


ROOT = Path('/kaggle/working/generative_recovery_2500')
STUDY = 'generative_recovery_1500'
CACHE_SHA = '98d39e0febb8c1946aba70eca2ca5cb96a657cdf6075f7ae32fc56d4c6f9bd5c'
SOURCE_NAMES = {'models.py','model_losses.py','compensated_interaction.py',
                'train_v3.py','interaction_witness.py','interaction_statistics.py'}


def require(value, message):
    if not value: raise ValueError(message)


def validate_config(config, root):
    root=Path(root)
    require(not root.is_symlink() and root.resolve()==ROOT,'Exact fresh v3 root required')
    fixed={'study_id':STUDY,'batch_size':64,'latent_dim':128,'base_channels':32,
        'learning_rate':.0002,'adam_betas':[0.,.9],'ema_decays':[.99,.999],
        'amp':True,'amp_initial_scale':1024,'cache_sha256':CACHE_SHA,
        'gsr_beta':.999,'shared_latent_g_batch':True,'teacher_feature_dim':128,
        'fake_ema_beta':.9,'witness_warmup_successful_updates':500,'scale_floor':.001,
        'pretrained_generator_or_discriminator':False,'quality_metrics_during_training':False}
    require(all(config.get(k)==v for k,v in fixed.items()),'V3 fixed training recipe differs')
    require(set(config['source_sha256'])==SOURCE_NAMES,'Exactly six v3 training source hashes required')
    for name,digest in config['source_sha256'].items():
        require(sha256_file(Path(__file__).parent/name)==digest,'Training source changed: '+name)
    require(config['phase'] in ('development','confirmation','pilot'),'Wrong phase')
    require(config['steps'] in (600,1500),'Wrong prospective attempt budget')
    require(config['seed'] in ([909] if config['phase']=='development' else
        [101,202,303] if config['phase']=='confirmation' else [707]),'Seed/phase differs')
    require(config['steps']==1500 if config['phase']=='development' else True,'Development must be1500')
    require(config['steps']==600 if config['phase']=='pilot' else True,'Active pilot must be600')
    require(config['phase']!='confirmation' or config['steps']==1500,'Confirmation fixed1500')
    require(config['arm'] in ('B','C'),'Wrong arm')
    require(config['coefficient'] in ([0.] if config['arm']=='B' else
        [.1,1.] if config['arm'] in ('P','C') else [.1,.5] if config['arm']=='GSR' else [1.]),'Coefficient not in frozen grid')
    require(config['kappa'] in (0.,1.),'Kappa not in frozen grid')
    require(config['arm']=='C' or config['kappa']==0.,'Kappa belongs only to C')
    require(config['run_id'] and all(c.isalnum() or c=='_' for c in config['run_id']),'Unsafe run identity')
    ready_path=root/'data/data_ready.json';ready=json.loads(ready_path.read_text())
    require(sha256_file(ready_path)==config['data_ready_sha256'] and
        ready['status']=='frozen_clean_component_split' and ready['study_id'] in ('generative_recovery_2500','generative_recovery_1500')
        and ready['test_quality_accessed'] is False,
        'Actual approved clean split not ready')
    require(config['metadata_npz']==str(root/'data/clean_index.npz') and
        config['component_ids_path']==str(root/'data/component_ids.npy'),'Wrong clean input path')
    for key,sha_key in [('metadata_npz','index_sha256'),('component_ids_path','component_ids_sha256'),
                        ('cache_path','cache_sha256')]:
        path=Path(config[key]);require(not path.is_symlink() and path.is_file(),'Missing/alias input '+key)
        require(sha256_file(path)==config[sha_key],'Data bytes changed '+key)
    require(ready['clean_index_sha256']==config['index_sha256'] and
        ready['component_ids_sha256']==config['component_ids_sha256'] and
        ready['original_cache_sha256']==CACHE_SHA,'Ready record/input binding differs')
    if config['phase']=='confirmation':
        decision=root/'selection_frozen.json'
        require(sha256_file(decision)==config['selection_sha256'],'Validation selection not frozen')
    if config['arm']=='C':
        for name in ['teacher_artifact','teacher_statistics']:
            path=Path(config[name+'_path'])
            require(path==root/'teacher'/('fixed_teacher.pt' if name=='teacher_artifact' else 'statistics.json')
                and not path.is_symlink() and sha256_file(path)==config[name+'_sha256'],'Fixed teacher changed')
    return True


def shared_latent_noise(batch_noise, labels):
    """Same16 latents in each condition; all64 draws still advance the RNG."""
    require(batch_noise.shape[0]==64,'Common batch64 required')
    ranks=torch.empty_like(labels)
    for group in range(4):
        mask=labels==group
        require(int(mask.sum())==16,'Balanced16/group required')
        ranks[mask]=torch.arange(16,device=labels.device)
    return batch_noise[:16][ranks]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Explicitly disclose that GPU kernel determinism is not guaranteed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class BalancedImageStream:
    def __init__(self, cache_path, group_labels, train_indices, seed, batch_size):
        if batch_size % 4 != 0:
            raise ValueError('Balanced batch size must be divisible by four')
        self.array = np.load(cache_path, mmap_mode='r')
        if self.array.shape != (202599,3,64,64) or self.array.dtype != np.uint8:
            raise ValueError('Cache must contain all official RGB64 uint8 images in filename order')
        self.groups = [np.asarray(train_indices)[np.asarray(group_labels)[train_indices]==g]
                       for g in range(4)]
        if not all(len(g)>0 for g in self.groups):
            raise ValueError('Every training group must be nonempty')
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.exposures = np.zeros(len(self.array),dtype=np.int64)

    def next(self):
        indices = np.concatenate([self.rng.choice(pool, self.batch_size//4, replace=True)
                                  for pool in self.groups])
        labels = np.repeat(np.arange(4),self.batch_size//4)
        order = self.rng.permutation(self.batch_size)
        indices, labels = indices[order], labels[order]
        np.add.at(self.exposures,indices,1)
        pixels = torch.from_numpy(np.array(self.array[indices],copy=True)).pin_memory()
        real = pixels.to('cuda',non_blocking=True).float().div_(127.5).sub_(1)
        return real, torch.from_numpy(labels).to('cuda'), indices


@torch.no_grad()
def update_ema(ema, model, decay):
    for p_ema, p_model in zip(ema.parameters(), model.parameters()):
        p_ema.lerp_(p_model,1-decay)
    for b_ema, b_model in zip(ema.buffers(),model.buffers()):
        b_ema.copy_(b_model)


def augment_flip(images, generator):
    mask = torch.rand((len(images),1,1,1), device=images.device,
                      generator=generator) < 0.5
    return torch.where(mask, images.flip(-1), images)


def rng_state(noise_rng, augment_rng, stream):
    return {'python':random.getstate(), 'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all(),
            'noise':noise_rng.get_state(), 'augmentation':augment_rng.get_state(),
            'sampler':stream.rng.bit_generator.state}


def train_one(config, root):
    validate_config(config, root)
    arm,seed=config['arm'],config['seed']
    pilot=config['phase']=='pilot'
    if arm == 'GSR':
        from model_losses import group_spectral_regularization
    if not torch.cuda.is_available():
        raise RuntimeError('This protocol requires an actual GPU runtime')
    torch.set_num_threads(config.get('cpu_threads',2))
    seed_everything(seed)
    run_dir = Path(root) / config['phase'] / config['run_id']
    run_dir.mkdir(parents=True,exist_ok=False)
    metadata = np.load(config['metadata_npz'],allow_pickle=False)
    labels = metadata['groups']; split = metadata['splits']
    train_indices = np.flatnonzero(split==0)
    require(labels.shape==split.shape==(202599,) and labels.dtype.kind in 'iu'
        and split.dtype.kind=='i' and np.isin(split,[-1,0,1,2]).all()
        and np.isin(labels,[0,1,2,3]).all(),'Invalid clean index')
    observed_counts=np.bincount(labels[train_indices],minlength=4).tolist()
    if observed_counts!=config['train_group_counts']:
        raise ValueError('Observed training group counts differ from the frozen configuration')
    stream = BalancedImageStream(config['cache_path'],labels,train_indices,seed+110,
                                 config['batch_size'])
    generator = Generator(z_dim=config['latent_dim'],num_classes=4,
                          base_channels=config['base_channels']).cuda()
    discriminator = ProjectionDiscriminator(num_classes=4,
                          base_channels=config['base_channels']).cuda()
    emas = {str(decay):copy.deepcopy(generator).eval().requires_grad_(False)
            for decay in config['ema_decays']}
    teacher=witness=None;teacher_before=None
    if arm=='C':
        from interaction_witness import LaggedInteractionWitness
        teacher=ProjectionDiscriminator(num_classes=4,base_channels=32).cuda()
        artifact=torch.load(config['teacher_artifact_path'],map_location='cpu',weights_only=True)
        require(artifact['study_id']==STUDY and artifact['arm']=='B' and artifact['seed']==909
            and artifact['step']==1500 and artifact['source_sha256']==config['source_sha256']
            and artifact['index_sha256']==config['index_sha256']
            and artifact['component_ids_sha256']==config['component_ids_sha256'],'Teacher provenance differs')
        teacher.load_state_dict(artifact['discriminator'],strict=True)
        teacher.eval().requires_grad_(False)
        teacher_before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
        statistics=json.loads(Path(config['teacher_statistics_path']).read_text())
        require(statistics['teacher_artifact_sha256']==config['teacher_artifact_sha256']
            and statistics['index_sha256']==config['index_sha256']
            and statistics['component_ids_sha256']==config['component_ids_sha256'], 'Statistics provenance differs')
        witness=LaggedInteractionWitness(statistics['statistics'],kappa=config['kappa'],
            beta=.9,warmup_updates=500).cuda()
    opt_g = torch.optim.Adam(generator.parameters(),lr=config['learning_rate'],
                             betas=tuple(config['adam_betas']))
    opt_d = torch.optim.Adam(discriminator.parameters(),lr=config['learning_rate'],
                             betas=tuple(config['adam_betas']))
    scaler_g = torch.amp.GradScaler('cuda',enabled=config['amp'],init_scale=config.get('amp_initial_scale',1024))
    scaler_d = torch.amp.GradScaler('cuda',enabled=config['amp'],init_scale=config.get('amp_initial_scale',1024))
    noise_rng = torch.Generator(device='cuda').manual_seed(seed+220)
    augment_rng = torch.Generator(device='cuda').manual_seed(seed+330)
    n_steps = config['steps']
    record = {'arm':arm,'seed':seed,'pilot':pilot,'config':config,
              'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'generator_parameters':sum(p.numel() for p in generator.parameters()),
              'discriminator_parameters':sum(p.numel() for p in discriminator.parameters()),
              'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
              'python':platform.python_version(),'numpy':np.__version__,
              'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
              'study_id':STUDY,'deterministic_gpu_kernels':False,
              'pretrained_generator_or_discriminator':False}
    record['teacher_auxiliary_parameters']=0 if teacher is None else sum(p.numel() for p in teacher.parameters())
    record['common_recipe_changed_from_v1']='shared16z_four_conditions_oneGforward'
    json_dump(run_dir/'start.json',record)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.monotonic(); measured_start = None
    skipped_g=skipped_d=0
    try:
        with (run_dir/'training.jsonl').open('a',buffering=1) as log:
            for step in range(1,n_steps+1):
                real,y,_ = stream.next()
                batch = len(y)
                generator.train(); discriminator.train().requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                z = torch.randn((batch,config['latent_dim']),device='cuda',generator=noise_rng)
                with torch.no_grad(), torch.autocast('cuda',enabled=config['amp'],dtype=torch.float16):
                    fake_d=generator(z,y)
                with torch.autocast('cuda',enabled=config['amp'],dtype=torch.float16):
                    real_scores=discriminator(augment_flip(real,augment_rng),y)
                    fake_scores=discriminator(augment_flip(fake_d,augment_rng),y)
                    loss_d=hinge_discriminator_loss(real_scores,fake_scores,y,y,num_classes=4)
                scale_d=scaler_d.get_scale()
                scaler_d.scale(loss_d).backward(); scaler_d.step(opt_d); scaler_d.update()
                skipped_d += int(scaler_d.get_scale()<scale_d)
                discriminator.eval().requires_grad_(False)
                opt_g.zero_grad(set_to_none=True)
                z = torch.randn((batch,config['latent_dim']),device='cuda',generator=noise_rng)
                with torch.autocast('cuda',enabled=config['amp'],dtype=torch.float16):
                    z=shared_latent_noise(z,y)
                    fake_g=generator(z,y)
                    generator_scores=discriminator(augment_flip(fake_g,augment_rng),y)
                    loss_g_adv=generator_adversarial_loss(generator_scores,y,
                                      weights=None,num_classes=4)
                # Affine penalties in FP32, outside autocast.
                penalty=coarse_interaction_penalty(generator,max_resolution=16)
                compensated_penalty=compensated_interaction_penalty(generator,max_resolution=16)
                loss_g=loss_g_adv.float()
                if arm=='V2':
                    loss_g=loss_g+config['coefficient']*penalty
                if arm=='P':
                    loss_g=loss_g+config['coefficient']*compensated_penalty
                if arm=='GSR':
                    loss_g=loss_g+group_spectral_regularization(generator,
                           config['train_group_counts'],beta=config['gsr_beta'],
                           regularizer_weight=config['coefficient'])
                witness_loss=loss_g.new_zeros(())
                diagnostics={};current_contrast=None
                if arm=='C':
                    # Frozen teacher weights; preserve gradient through fake_g INPUT.
                    with torch.autocast('cuda',enabled=False):
                        _,features=teacher(fake_g.float(),y,return_features=True)
                        witness_loss,diagnostics,current_contrast=witness.loss(
                            features,y,strength=config['coefficient'])
                        require(features.requires_grad and witness_loss.requires_grad,
                            'Teacher input gradient to G was detached')
                    loss_g=loss_g+witness_loss
                scale_g=scaler_g.get_scale()
                scaler_g.scale(loss_g).backward(); scaler_g.step(opt_g); scaler_g.update()
                generator_step_skipped=scaler_g.get_scale()<scale_g
                skipped_g += int(generator_step_skipped)
                if not generator_step_skipped:
                    for decay,ema in emas.items():update_ema(ema,generator,float(decay))
                    if witness is not None:witness.update(current_contrast)
                if step==config['timing_warmup']:
                    torch.cuda.synchronize(); measured_start=time.monotonic()
                if step==1 or step%config['log_every']==0 or step==n_steps:
                    torch.cuda.synchronize()
                    entry={'step':step,'real_images':step*batch,
                           'loss_d':float(loss_d.detach()),'loss_g':float(loss_g.detach()),
                           'interaction_penalty':float(penalty.detach()),
                           'compensated_penalty':float(compensated_penalty.detach()),
                           'witness_loss_linear_surrogate':float(witness_loss.detach()),
                           'witness_diagnostics':{k:float(v.detach()) for k,v in diagnostics.items()},
                           'applied_group_weights':[.25]*4,
                           'elapsed_s':time.monotonic()-start,
                           'max_allocated_bytes':torch.cuda.max_memory_allocated(),
                           'amp_skipped_g':skipped_g,'amp_skipped_d':skipped_d}
                    if not all(math.isfinite(entry[k]) for k in ('loss_d','loss_g','interaction_penalty','compensated_penalty')):
                        raise FloatingPointError(f'Non-finite training loss at step {step}')
                    log.write(json.dumps(entry)+'\n')
                    print(json.dumps({'run':run_dir.name,**entry}),flush=True)
                if not pilot and (step%config['checkpoint_every']==0 or step==n_steps):
                    checkpoint={'generator':generator.state_dict(),'ema_generator':emas['0.999'].state_dict(),
                         'ema_generators':{decay:model.state_dict() for decay,model in emas.items()},
                         'discriminator':discriminator.state_dict(),'config':config,'step':step,
                         'seed':seed,'arm':arm,'optimizer_g':opt_g.state_dict(),
                         'optimizer_d':opt_d.state_dict(),'scaler_g':scaler_g.state_dict(),
                         'scaler_d':scaler_d.state_dict(),'rng':rng_state(noise_rng,augment_rng,stream),
                         'witness':None if witness is None else witness.state_dict(),
                         'witness_config':None if witness is None else witness.config(),
                         'exposures':stream.exposures}
                    temporary=run_dir/'latest.tmp.pt'
                    torch.save(checkpoint,temporary); temporary.replace(run_dir/'latest.pt')
                    np.save(run_dir/'exposures.npy',stream.exposures)
            torch.cuda.synchronize()
            seconds=time.monotonic()-start
            measured_seconds=(time.monotonic()-measured_start) if measured_start else seconds
            measured_steps=n_steps-config['timing_warmup'] if measured_start else n_steps
            exposure_by_group=[int(stream.exposures[pool].sum()) for pool in stream.groups]
            unique_by_group=[int(np.count_nonzero(stream.exposures[pool])) for pool in stream.groups]
            result={**record,'status':'complete','completed_steps':n_steps,
                    'real_images':n_steps*config['batch_size'],'elapsed_s':seconds,
                    'measured_seconds_per_step':measured_seconds/max(measured_steps,1),
                    'max_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'max_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'group_exposures':exposure_by_group,'group_unique_images':unique_by_group,
                    'amp_skipped_g':skipped_g,'amp_skipped_d':skipped_d,
                    'successful_g_updates':n_steps-skipped_g,
                    'successful_d_updates':n_steps-skipped_d}
            if not pilot:
                result['checkpoint_sha256']=sha256_file(run_dir/'latest.pt')
                d_export={'study_id':STUDY,'discriminator':{k:v.detach().cpu() for k,v in discriminator.state_dict().items()},
                    'index_sha256':config['index_sha256'],'component_ids_sha256':config['component_ids_sha256'],
                    'base_channels':32,'seed':seed,'step':n_steps,'arm':arm,'source_sha256':config['source_sha256']}
                torch.save(d_export,run_dir/'D_final.pt')
                result['D_final_sha256']=sha256_file(run_dir/'D_final.pt')
                result['ema_artifacts']={}
                for decay,ema in emas.items():
                    path=run_dir/('EMA_'+decay+'.pt')
                    torch.save({'ema_generator':{k:v.detach().cpu() for k,v in ema.state_dict().items()},
                        'generator_config':generator.config(),'study_id':STUDY,'step':n_steps,'arm':arm,'seed':seed,
                        'ema_decay':float(decay),'source_sha256':config['source_sha256'],
                        'config':config},path)
                    result['ema_artifacts'][decay]={'path':str(path),'sha256':sha256_file(path)}
            result['witness_updates']=None if witness is None else int(witness.updates.item())
            if arm=='C':
                require(all(torch.equal(value,teacher_before[key]) for key,value in teacher.state_dict().items()),
                    'Fixed teacher parameters or SN buffers changed')
                require(all(parameter.grad is None for parameter in teacher.parameters()),
                    'Fixed teacher accumulated parameter gradients')
                result['teacher_state_unchanged']=True
                require(sha256_file(config['teacher_artifact_path'])==config['teacher_artifact_sha256']
                    and sha256_file(config['teacher_statistics_path'])==config['teacher_statistics_sha256'],
                    'Fixed teacher changed during run')
            json_dump(run_dir/'complete.json',result)
            print('RUN_COMPLETE '+json.dumps(result),flush=True)
            return result
    except BaseException as error:
        json_dump(run_dir/'failure.json',{'type':type(error).__name__,'error':str(error),
                       'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-start})
        raise
    finally:
        del generator,discriminator,emas,opt_g,opt_d,teacher,witness
        torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--root',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    try:train_one(config,args.root)
    except BaseException as error:
        failures=Path(args.root)/'failures';failures.mkdir(parents=True,exist_ok=True)
        target=failures/(config['run_id']+'.json')
        if not target.exists():json_dump(target,{'type':type(error).__name__,'error':str(error),
            'traceback':traceback.format_exc(),'config':config})
        raise
