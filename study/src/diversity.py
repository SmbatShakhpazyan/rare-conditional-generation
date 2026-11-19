"""Descriptive PRDC, finite-reference nearest distances and pixel duplicates.

Fresh NumPy implementation of Naeem et al. ICML2020 PRDC definitions, checked
against clovaai/generative-evaluation-prdc/prdc/prdc.py (MIT, NAVER Corp.).
No framework, model, download or actual feature dataset is loaded on import.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


PRDC_SOURCE = "https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py"
PRDC_VERSION = "numpy_full_finite_sets_float64_strict_balls_v1"


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _features(value, name):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.dtype.kind not in "fiu" or min(raw.shape) < 1:
        raise ValueError(name + " must be a nonempty real N-by-d feature matrix")
    values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(name + " contains nonfinite features")
    return values


def _distance_block(x, y, feature_block_size):
    # Direct differences avoid cancellation in ||x||^2+||y||^2-2<x,y>.
    # Feature chunks bound memory even for a large nearest-training bank.
    squared = np.zeros((len(x), len(y)), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        for start in range(0, x.shape[1], feature_block_size):
            delta = x[:, None, start:start + feature_block_size] - y[None, :, start:start + feature_block_size]
            squared += np.einsum("ijk,ijk->ij", delta, delta, optimize=False)
        np.sqrt(squared, out=squared)
    if not np.isfinite(squared).all():
        raise FloatingPointError("Nonfinite Euclidean distance")
    return squared


def pairwise_euclidean(x, y=None, row_block_size=32, column_block_size=256,
                       feature_block_size=128, max_pairs=4194304):
    """All finite-set distances, not approximate neighbors; arithmetic is float64.

    max_pairs is an explicit dense-memory gate, not silent data subsampling.
    For a large reference bank use nearest_reference_distances instead.
    """
    x = _features(x, "x")
    y = x if y is None else _features(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimensions differ")
    row_block_size = _positive_int(row_block_size, "row_block_size")
    column_block_size = _positive_int(column_block_size, "column_block_size")
    feature_block_size = _positive_int(feature_block_size, "feature_block_size")
    max_pairs = _positive_int(max_pairs, "max_pairs")
    if len(x) * len(y) > max_pairs:
        raise ValueError("Dense pair limit exceeded; declare an outcome-free subset or stream nearest distances")
    result = np.empty((len(x), len(y)), dtype=np.float64)
    for i in range(0, len(x), row_block_size):
        for j in range(0, len(y), column_block_size):
            result[i:i + row_block_size, j:j + column_block_size] = _distance_block(
                x[i:i + row_block_size], y[j:j + column_block_size], feature_block_size)
    return result


def nearest_neighbour_radii(features, nearest_k=3, **distance_kwargs):
    features = _features(features, "features")
    nearest_k = _positive_int(nearest_k, "nearest_k")
    if len(features) <= nearest_k:
        raise ValueError("Need at least k+1 observations for kth-other-neighbor radii")
    distances = pairwise_euclidean(features, **distance_kwargs)
    # The diagonal contributes one self-zero; coincident OTHER observations
    # remain valid zero-distance neighbors. Sorting index k excludes one self.
    return np.partition(distances, nearest_k, axis=1)[:, nearest_k]


def compute_prdc(real_features, fake_features, nearest_k=3, **distance_kwargs):
    """Strict-ball PRDC on exactly the supplied sets, allowing unequal counts.

    Precision/recall/coverage are fractions. Density can exceed one. Duplicate
    observations/ties are retained; zero-radius balls are empty under '<'.
    """
    real = _features(real_features, "real")
    fake = _features(fake_features, "fake")
    if real.shape[1] != fake.shape[1]:
        raise ValueError("Feature dimensions differ")
    nearest_k = _positive_int(nearest_k, "nearest_k")
    real_radius = nearest_neighbour_radii(real, nearest_k, **distance_kwargs)
    fake_radius = nearest_neighbour_radii(fake, nearest_k, **distance_kwargs)
    distances = pairwise_euclidean(real, fake, **distance_kwargs)
    inside_real = distances < real_radius[:, None]
    inside_fake = distances < fake_radius[None, :]
    return {
        "precision": float(inside_real.any(axis=0).mean()),
        "recall": float(inside_fake.any(axis=1).mean()),
        "density": float(inside_real.sum(dtype=np.int64) / (len(fake) * nearest_k)),
        "coverage": float(inside_real.any(axis=1).mean()),
        "n_real": len(real), "n_fake": len(fake), "nearest_k": nearest_k,
        "zero_real_radius_count": int(np.count_nonzero(real_radius == 0)),
        "zero_fake_radius_count": int(np.count_nonzero(fake_radius == 0)),
        "strict_ball_membership": True, "density_is_probability": False,
        "method": PRDC_VERSION, "primary_definition_source": PRDC_SOURCE,
    }


def evaluate_prdc_groups(real, fake, nearest_k=3, analysis_cap=512,
                         selection_seed=1701, kid_point_report=None, **distance_kwargs):
    """Reuse the KID evaluator's exact ID-hash-selected requested-bin subsets.

    Optional saved KID point report is checked for identical selected IDs;
    no KID or model inference is recomputed by this function.
    """
    try:
        from .evaluate import _checked_dataset, _selected_group, GROUPS
    except ImportError:
        from evaluate import _checked_dataset, _selected_group, GROUPS
    real, fake = _checked_dataset(real, "real"), _checked_dataset(fake, "fake")
    nearest_k = _positive_int(nearest_k, "nearest_k")
    if analysis_cap is not None:
        analysis_cap = _positive_int(analysis_cap, "analysis_cap")
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, (int, np.integer)) or selection_seed < 0:
        raise ValueError("selection_seed must be a nonnegative integer")
    extractor = real["provenance"]["feature_extractor"]
    if extractor != fake["provenance"]["feature_extractor"]:
        raise ValueError("Feature extractor/weights/preprocessing provenance differs")
    if kid_point_report is not None and kid_point_report.get("feature_definition") != extractor:
        raise ValueError("PRDC feature definition differs from the saved point-KID report")
    results = {}
    for group in GROUPS:
        x, xid = _selected_group(real, group, analysis_cap, selection_seed)
        y, yid = _selected_group(fake, group, analysis_cap, selection_seed)
        if kid_point_report is not None:
            expected = kid_point_report["groups"][str(group)]
            if xid != expected["real_ids"] or yid != expected["fake_requested_ids"]:
                raise ValueError("PRDC point subsets differ from saved point-KID subsets")
        results[str(group)] = {**compute_prdc(x, y, nearest_k, **distance_kwargs),
                               "real_ids": xid, "fake_requested_ids": yid}
    return {
        "status": "computed_descriptive", "metric_version": PRDC_VERSION,
        "nearest_k": nearest_k, "analysis_cap": analysis_cap, "selection_seed": int(selection_seed),
        "point_subset_matches_supplied_KID_report": kid_point_report is not None,
        "requested_bins_used": True, "generated_filtering": False, "generated_relabeling": False,
        "feature_definition": extractor, "groups": results,
        "macro": {metric: float(np.mean([results[str(g)][metric] for g in GROUPS]))
                  for metric in ["precision", "recall", "density", "coverage"]},
        "scope": "Finite-feature descriptive diagnostics, not semantic correctness or a population guarantee.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
    }


def _ids(values, count, name):
    values = np.asarray(values)
    if values.shape != (count,) or values.dtype.kind not in "US":
        raise ValueError(name + " must contain one string ID per row")
    values = values.astype(str)
    if any(not item for item in values) or len(set(values.tolist())) != count:
        raise ValueError(name + " must be nonempty and unique")
    return values


def nearest_reference_distances(query_features, reference_features, query_ids=None,
                                reference_ids=None, exclude_matching_ids=False,
                                query_block_size=32, reference_block_size=256,
                                feature_block_size=128):
    """Exact search over the supplied finite bank, without a dense Q-by-R array.

    Lowest input reference index breaks an exact tie. To use a real training
    query against its own bank, set exclude_matching_ids=True and supply IDs.
    Feature equality/distance alone neither proves nor rules out memorization.
    """
    query, reference = _features(query_features, "query"), _features(reference_features, "reference")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("Feature dimensions differ")
    query_block_size = _positive_int(query_block_size, "query_block_size")
    reference_block_size = _positive_int(reference_block_size, "reference_block_size")
    feature_block_size = _positive_int(feature_block_size, "feature_block_size")
    qids = None if query_ids is None else _ids(query_ids, len(query), "query_ids")
    rids = None if reference_ids is None else _ids(reference_ids, len(reference), "reference_ids")
    if exclude_matching_ids and (qids is None or rids is None):
        raise ValueError("Matching-ID exclusion needs both query and reference IDs")
    distances = np.full(len(query), np.inf)
    indices = np.full(len(query), -1, dtype=np.int64)
    excluded = 0
    for i in range(0, len(query), query_block_size):
        q = query[i:i + query_block_size]
        best, best_index = distances[i:i + len(q)], indices[i:i + len(q)]
        for j in range(0, len(reference), reference_block_size):
            block = _distance_block(q, reference[j:j + reference_block_size], feature_block_size)
            if exclude_matching_ids:
                mask = qids[i:i + len(q), None] == rids[None, j:j + block.shape[1]]
                excluded += int(mask.sum())
                block[mask] = np.inf
            local_index = np.argmin(block, axis=1)
            local_best = block[np.arange(len(q)), local_index]
            improve = local_best < best
            best[improve] = local_best[improve]
            best_index[improve] = j + local_index[improve]
    if not np.isfinite(distances).all() or (indices < 0).any():
        raise ValueError("At least one query has no admissible finite reference")
    return {
        "distances": distances, "reference_indices": indices,
        "nearest_reference_ids": None if rids is None else rids[indices].tolist(),
        "summary": {"n_query": len(query), "n_reference": len(reference),
                    "minimum": float(distances.min()), "mean": float(distances.mean()),
                    "median": float(np.median(distances)), "p95": float(np.quantile(distances, .95)),
                    "maximum": float(distances.max()),
                    "exact_feature_match_fraction": float(np.mean(distances == 0)),
                    "excluded_matching_ID_pairs": excluded},
        "method": "full_supplied_finite_bank_float64_direct_Euclidean_chunked",
        "approximate_neighbors": False, "feature_normalization_applied": False,
        "distance_is_memorization_certificate": False,
    }


def _duplicate_summary(counts):
    total, unique = sum(counts), len(counts)
    return {"n_images": total, "n_unique_pixel_arrays": unique,
            "duplicate_fraction_after_first_copy": (total - unique) / total,
            "fraction_in_repeated_clusters": sum(n for n in counts if n > 1) / total,
            "exact_duplicate_pairs": sum(n * (n - 1) // 2 for n in counts),
            "maximum_multiplicity": max(counts)}


def pixel_duplicate_report(images, groups=None):
    """Exact duplicates AFTER the caller's common canonical uint8 quantization.

    SHA256 locates candidate equal arrays; byte equality is then verified.
    No approximate perceptual duplicate/privacy claim and no image filtering.
    """
    pixels = np.asarray(images)
    if (pixels.dtype != np.uint8 or pixels.ndim != 4 or len(pixels) < 1
            or pixels.shape[-1] != 3 or min(pixels.shape[1:3]) < 1):
        raise ValueError("Need nonempty canonical uint8 BHWC RGB images")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(pixels),) or groups.dtype.kind not in "iu":
            raise ValueError("Need an integer requested-group label per image")
    clusters, per_group = {}, {}
    for index, image in enumerate(pixels):
        digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()
        if digest in clusters:
            first, count = clusters[digest]
            if not np.array_equal(pixels[first], image):
                raise RuntimeError("SHA256 collision among unequal image arrays")
            clusters[digest] = (first, count + 1)
        else:
            clusters[digest] = (index, 1)
        if groups is not None:
            counter = per_group.setdefault(int(groups[index]), {})
            counter[digest] = counter.get(digest, 0) + 1
    return {
        "overall": _duplicate_summary([item[1] for item in clusters.values()]),
        "per_requested_group": {str(g): _duplicate_summary(list(counter.values()))
                                for g, counter in sorted(per_group.items())},
        "pixel_shape": list(pixels.shape[1:]), "canonical_dtype": "uint8",
        "hash_candidates_verified_by_pixel_equality": True,
        "filtering": False, "perceptual_duplicates_measured": False,
        "not_training_image_membership_or_privacy_certificate": True,
    }
