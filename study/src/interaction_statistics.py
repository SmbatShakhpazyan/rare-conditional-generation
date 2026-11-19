"""Train-only, image-weighted signed moments with component-cluster scales.

New elementary statistics implementation for the v2 proposed combination.
No Torch, data access, identity download, or computation occurs on import.
The uncertainty scale is a cluster-sandwich HEURISTIC, not a confidence bound.
"""
import math

import numpy as np


CONTRAST = (0.25, -0.25, -0.25, 0.25)


def compute_real_statistics(features, groups, cluster_ids, scale_floor=1e-3):
    """Return JSON-compatible fixed-teacher statistics; do not use val/test.

    The mean in each group remains image weighted. The cluster contribution
    sums all groups for each identity BEFORE squaring, retaining cross-group
    covariance. Repeated images/presentations must not be passed as fresh data.
    Input cluster IDs are union components of the leakage audit (identity, exact,
    flip and specified near duplicates). They are used only in memory; returned
    records contain counts, not IDs. At least two components/group required.
    """
    if isinstance(scale_floor, bool) or not math.isfinite(float(scale_floor)) or scale_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    x = np.asarray(features)
    g = np.asarray(groups)
    ids = np.asarray(cluster_ids)
    if x.ndim != 2 or min(x.shape) < 1 or not np.issubdtype(x.dtype, np.number):
        raise ValueError("features must be a nonempty numeric [N,d] array")
    if np.issubdtype(x.dtype, np.complexfloating) or np.issubdtype(x.dtype, np.bool_):
        raise ValueError("features must be real valued")
    if g.shape != (len(x),) or ids.shape != (len(x),):
        raise ValueError("groups/cluster_ids must have length N")
    if not np.issubdtype(g.dtype, np.integer) or np.issubdtype(g.dtype, np.bool_):
        raise ValueError("groups must contain integer annotation bins")
    if not np.issubdtype(ids.dtype, np.integer) or np.issubdtype(ids.dtype, np.bool_):
        raise ValueError("cluster_ids must contain integer leakage-component IDs")
    if np.any(g < 0) or np.any(g > 3):
        raise ValueError("groups must be 0,1,2,3 in 2*Male+Blond_Hair order")
    x = x.astype(np.float64, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite features")
    _, inverse = np.unique(ids, return_inverse=True)
    n_identities = int(inverse.max()) + 1
    if n_identities < 2:
        raise ValueError("at least two leakage-component clusters required")
    d = x.shape[1]
    cluster_influence = np.zeros((n_identities, d), dtype=np.float64)
    means = []
    pooled_variance = np.zeros(d, dtype=np.float64)
    group_counts = []
    identity_counts = []
    effective_counts = []
    for group, coefficient in enumerate(CONTRAST):
        selected = g == group
        n = int(selected.sum())
        if n < 2:
            raise ValueError("every group needs at least two images")
        counts = np.bincount(inverse[selected], minlength=n_identities)
        n_clusters = int(np.count_nonzero(counts))
        if n_clusters < 2:
            raise ValueError("every group needs at least two distinct leakage components")
        mu = x[selected].mean(axis=0)
        centered = x[selected] - mu
        means.append(mu)
        pooled_variance += np.mean(centered * centered, axis=0) / 4.0
        np.add.at(cluster_influence, inverse[selected], centered * (coefficient / n))
        group_counts.append(n)
        identity_counts.append(n_clusters)
        effective_counts.append(float(n * n / np.sum(counts.astype(np.float64) ** 2)))
    target = np.asarray(CONTRAST, dtype=np.float64) @ np.asarray(means)
    variance = (n_identities / (n_identities - 1.0)) * np.sum(cluster_influence ** 2, axis=0)
    scale = np.maximum(np.sqrt(pooled_variance), float(scale_floor))
    uncertainty = np.sqrt(variance)
    if not all(np.isfinite(a).all() for a in (target, scale, uncertainty)):
        raise ValueError("nonfinite derived statistics")
    return {
        "target": target.tolist(), "scale": scale.tolist(),
        "uncertainty": uncertainty.tolist(), "group_means": np.asarray(means).tolist(),
        "contrast": list(CONTRAST), "feature_dim": int(d), "n_images": int(len(x)),
        "group_image_counts": group_counts, "n_clusters": n_identities,
        "group_cluster_counts": identity_counts,
        "cluster_weight_effective_counts": effective_counts,
        "scale_floor": float(scale_floor),
        "target_weighting": "image_weighted_within_each_of_four_equal_weight_groups",
        "uncertainty_method": "diagonal_leakage_component_sandwich_M_over_Mminus1",
        "cross_group_cluster_covariance_included": True,
        "confidence_bound": False,
        "independent_information_is_not_repeated_presentations": True,
    }


# Descriptive backwards-compatible alias; input IDs must still be leakage components.
real_interaction_statistics = compute_real_statistics
