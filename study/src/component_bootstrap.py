"""Conditional fixed-network cubic-KID differences: component/latent bootstrap.
No learned models, dataset access, RNG draws or work occur on import.
"""
from __future__ import annotations
import time
import numpy as np


def require(value,message):
    if not value:raise ValueError(message)


def kernel(a,b):return (a@b.T/a.shape[1]+1.)**3


def _checked(value,real=False):
    features=np.asarray(value['features'],dtype=np.float64)
    groups=np.asarray(value['groups']);ids=np.asarray(value['ids']).astype(str)
    require(features.ndim==2 and features.shape[1]>0 and groups.ndim==ids.ndim==1
        and len(features)==len(groups)==len(ids) and np.isfinite(features).all(),'invalid_feature_rows')
    require(groups.dtype.kind in 'iu' and np.isin(groups,[0,1,2,3]).all()
        and len(set(ids.tolist()))==len(ids),'invalid_group_or_sample_IDs')
    if real:
        clusters=np.asarray(value['component_ids'])
        require(clusters.shape==groups.shape and clusters.dtype.kind in 'iu','real_component_IDs_required')
    else:
        latent=np.asarray(value['latent_indices'])
        require(latent.shape==groups.shape and latent.dtype.kind in 'iu','fake_latent_indices_required')
    return features,groups,ids


def paired_matrices(real,candidates,references):
    require(len(candidates)==len(references)==3,'exactly3_fixed_seed_pairs_required')
    x,xg,xids=_checked(real,True);parts={}
    checked=[(_checked(a),_checked(b)) for a,b in zip(candidates,references)]
    canonical=None
    for (a,ag,ai),(b,bg,bi) in checked:
        require(np.array_equal(ag,bg) and np.array_equal(ai,bi),'paired_model_IDs_differ')
        current=(ag,ai)
        if canonical is None:canonical=current
        else:require(np.array_equal(ag,canonical[0]) and np.array_equal(ai,canonical[1]),'seed_pair_IDs_differ')
    common_latent=None
    for group in range(4):
        ri=np.flatnonzero(xg==group);require(len(ri)>=2,'insufficient_real_rows')
        ca=[];cb=[];yids=None;latent=None
        for pair,(a,b) in enumerate(checked):
            ia=np.flatnonzero(a[1]==group);ib=np.flatnonzero(b[1]==group)
            require(np.array_equal(a[2][ia],b[2][ib]),'paired_group_IDs_differ')
            if yids is None:yids=a[2][ia]
            else:require(np.array_equal(yids,a[2][ia]),'fixed_seed_IDs_differ')
            la=np.asarray(candidates[pair]['latent_indices'])[ia]
            lb=np.asarray(references[pair]['latent_indices'])[ib]
            require(np.array_equal(la,lb),'paired_latent_indices_differ')
            if latent is None:latent=la
            else:require(np.array_equal(latent,la),'fixed_seed_latent_indices_differ')
            ca.append(a[0][ia]);cb.append(b[0][ib])
        require(len(latent)>=2 and np.array_equal(latent,np.arange(len(latent))),
            'fake_rows_must_be_in_complete_latent_index_order')
        if common_latent is None:common_latent=latent
        else:require(np.array_equal(common_latent,latent),'latent_coupling_across_groups_differ')
        yy=sum(kernel(a,a)-kernel(b,b) for a,b in zip(ca,cb))/3.
        xy=sum(kernel(x[ri],a)-kernel(x[ri],b) for a,b in zip(ca,cb))/3.
        parts[group]={'yy':yy,'xy':xy,'real_indices':ri,'real_ids':xids[ri].tolist(),
            'fake_ids':yids.tolist(),'latent_indices':latent.tolist()}
    return parts


def weighted_difference(yy,xy,real_counts,fake_counts):
    r=np.asarray(real_counts,dtype=np.float64);f=np.asarray(fake_counts,dtype=np.float64)
    if r.ndim==1:r=r[None,:]
    if f.ndim==1:f=f[None,:]
    require(r.ndim==f.ndim==2 and len(r)==len(f) and r.shape[1]==xy.shape[0]
        and f.shape[1]==xy.shape[1] and yy.shape==(f.shape[1],f.shape[1]),'weight_or_Gram_shape')
    require(np.isfinite(r).all() and np.isfinite(f).all() and (r>=0).all() and (f>=0).all()
        and np.isfinite(yy).all() and np.isfinite(xy).all(),'nonfinite_or_negative_weights_or_Gram')
    m=r.sum(axis=1);n=f.sum(axis=1)
    require(np.all(m>=2) and np.all(n>=2),'bootstrap_group_has_fewer_than2_rows_no_redraw')
    return (((f@yy)*f).sum(axis=1)-f@np.diag(yy))/(n*(n-1))-2*((r@xy)*f).sum(axis=1)/(m*n)


def component_paired_bootstrap(real,candidates,references,replicates=2000,seed=95003,
        confidence=.9875,max_seconds=600,batch_size=32):
    require(replicates>=2 and batch_size>=1 and .5<confidence<1 and max_seconds>0,'bootstrap_settings')
    began=time.monotonic();parts=paired_matrices(real,candidates,references)
    clusters=np.asarray(real['component_ids']);unique,inverse=np.unique(clusters,return_inverse=True)
    require(len(unique)>=2,'fewer_than2_real_components')
    n=len(parts[0]['latent_indices']);rng=np.random.default_rng(seed)
    scores=[];completed=0
    point=np.asarray([weighted_difference(p['yy'],p['xy'],np.ones(len(p['real_indices'])),np.ones(n))[0]
        for p in parts.values()])
    for start in range(0,replicates,batch_size):
        if time.monotonic()-began>=max_seconds:
            return {'status':'INCOMPLETE','reason':'bootstrap_soft_time_cap','completed_replicates':completed,
                'planned_replicates':replicates,'elapsed_seconds':time.monotonic()-began}
        count=min(batch_size,replicates-start)
        component_counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=count)
        fake_counts=rng.multinomial(n,np.full(n,1/n),size=count)
        batch=[]
        for group,p in parts.items():
            real_counts=component_counts[:,inverse[p['real_indices']]]
            if np.any(real_counts.sum(axis=1)<2):
                return {'status':'INCOMPLETE','reason':'component_resample_has_group_with_fewer_than2_rows_no_redraw',
                    'completed_replicates':completed,'planned_replicates':replicates}
            batch.append(weighted_difference(p['yy'],p['xy'],real_counts,fake_counts))
        scores.append(np.stack(batch,axis=1));completed+=count
    if time.monotonic()-began>=max_seconds:
        return {'status':'INCOMPLETE','reason':'bootstrap_soft_time_cap','completed_replicates':completed,
            'planned_replicates':replicates,'elapsed_seconds':time.monotonic()-began}
    draws=np.concatenate(scores,axis=0)
    def endpoint(values,estimate):
        return {'point_difference':float(estimate),'one_sided_upper':float(np.quantile(values,confidence,method='linear')),
            'two_sided_percentile':[float(v) for v in np.quantile(values,[(1-confidence)/2,1-(1-confidence)/2],method='linear')]}
    return {'status':'computed','method':'paired_leakage_component_and_shared_latent_bootstrap_mean3_fixed_seed_pairs',
        'replicates':replicates,'seed':seed,'batch_size':batch_size,
        'confidence_one_sided':confidence,'fixed_seed_pairs':3,
        'seed_resampling':False,'real_resampling':'union_of_observed_selected_reference_components_shared_all_groups',
        'fake_resampling':'common_latent_index_shared_all_groups_and_fixed_models','real_components':len(unique),
        'fake_latent_indices':n,'elapsed_seconds':time.monotonic()-began,
        'groups':{str(g):endpoint(draws[:,g],point[g]) for g in range(4)},
        'rare_group_primary':endpoint(draws[:,3],point[3]),
        'macro_descriptive':endpoint(draws.mean(axis=1),point.mean()),
        'multiplicity_scope':'rare_group_only_at_most4_predeclared_contrasts_one_sided_Bonferroni',
        'scope':'conditional_on_selected_finite_reference_bank_declared_components_and3_fixed_seed_pairs_not_training_population',
        'selected_ids':{str(g):{'real':p['real_ids'],'fake':p['fake_ids']} for g,p in parts.items()}}
