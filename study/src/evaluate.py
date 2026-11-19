"""Conditional image metrics without importing a training framework at import time.

The NumPy KID estimator uses unequal sample sizes and excludes self diagonals.
The optional torchvision adapter is explicitly NOT the Clean-FID feature model.
No network access, weight download, training, or cloud action occurs on import.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse

import numpy as np


GROUPS = (0, 1, 2, 3)
GROUP_LABELS = {
    0: {"Male": 0, "Blond_Hair": 0},
    1: {"Male": 0, "Blond_Hair": 1},
    2: {"Male": 1, "Blond_Hair": 0},
    3: {"Male": 1, "Blond_Hair": 1},
}
METRIC_VERSION = "conditional-kid-v1"


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _matrix(value, name, min_rows=2):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2:
        raise ValueError(f"{name} must be a real numeric N-by-d matrix")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape[0] < min_rows or array.shape[1] < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} needs >={min_rows} finite rows and >=1 feature")
    return array


def _kernel(x, y):
    with np.errstate(over="raise", invalid="raise"):
        result = (x @ y.T / x.shape[1] + 1.0) ** 3
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite cubic kernel; check feature scale/provenance")
    return result


def _kernel_sum(x, y, block_size, exclude_self=False):
    total = 0.0
    for i in range(0, len(x), block_size):
        for j in range(0, len(y), block_size):
            block = _kernel(x[i:i + block_size], y[j:j + block_size])
            if exclude_self and i == j:
                np.fill_diagonal(block, 0.0)
            total += float(block.sum(dtype=np.float64))
    return total


def polynomial_kid(real, fake, block_size=256):
    """Unbiased squared MMD for k(x,y)=(x.y/d+1)^3; negative values are valid.

    Unbiasedness is for independent samples in a fixed feature space. It does
    not imply that a worst-group maximum, bootstrap interval, or semantic
    fairness assessment is unbiased. Counts may differ; neither set is padded.
    """
    x, y = _matrix(real, "real"), _matrix(fake, "fake")
    block_size = _positive_int(block_size, "block_size")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimensions differ")
    n, m = len(x), len(y)
    result = float(
        _kernel_sum(x, x, block_size, True) / (n * (n - 1))
        + _kernel_sum(y, y, block_size, True) / (m * (m - 1))
        - 2 * _kernel_sum(x, y, block_size) / (n * m)
    )
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite KID; check feature scale")
    return result


def deterministic_latents(seed, group, count, latent_dim=128):
    """A group-specific PCG64 stream, independent of batching and other groups."""
    seed = _positive_int(seed, "seed", 0)
    group = _positive_int(group, "group", 0)
    if group not in GROUPS:
        raise ValueError("group must be0,1,2,3 (=2*Male+Blond_Hair)")
    count = _positive_int(count, "count")
    latent_dim = _positive_int(latent_dim, "latent_dim")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, group, 80317])))
    return rng.standard_normal((count, latent_dim), dtype=np.float32)


def generation_inputs(seed, count_per_group, latent_dim=128, batch_size=64):
    """Yield every requested sample once. No classifier, rejection or relabeling."""
    batch_size = _positive_int(batch_size, "batch_size")
    for group in GROUPS:
        z = deterministic_latents(seed, group, count_per_group, latent_dim)
        for start in range(0, len(z), batch_size):
            stop = min(start + batch_size, len(z))
            yield {
                "latents": z[start:stop],
                "groups": np.full(stop - start, group, dtype=np.int64),
                "ids": np.asarray([f"z:{seed}:{group}:{i}" for i in range(start, stop)]),
            }


def rgb_uint8(images):
    """Canonical quantization of BHWC RGB: tanh range -> round((x+1)*127.5)."""
    x = np.asarray(images)
    if x.ndim != 4 or x.shape[-1] != 3 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError("Images must be B-by-H-by-W-by3 RGB")
    if x.dtype == np.uint8:
        return x
    if x.dtype.kind != "f" or not np.isfinite(x).all() or np.any(x < -1) or np.any(x > 1):
        raise ValueError("Float images must be finite and in[-1,1]; no silent clipping")
    return np.rint((x.astype(np.float64) + 1.0) * 127.5).astype(np.uint8)


def _checked_dataset(dataset, name):
    features = _matrix(dataset["features"], name + ".features")
    groups = np.asarray(dataset["groups"])
    ids = np.asarray(dataset["ids"])
    if groups.ndim != 1 or groups.dtype.kind not in "iu" or len(groups) != len(features):
        raise ValueError(name + " has invalid group IDs")
    if not np.isin(groups, GROUPS).all():
        raise ValueError(name + " group IDs must be0..3")
    if ids.ndim != 1 or len(ids) != len(features) or ids.dtype.kind not in "US":
        raise ValueError(name + " needs a string ID for each feature row")
    ids = ids.astype(str)
    if any(not value for value in ids.tolist()) or len(set(ids.tolist())) != len(ids):
        raise ValueError(name + " sample IDs must be nonempty and unique")
    provenance = dataset.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("feature_extractor"), dict):
        raise ValueError(name + " needs feature_extractor provenance")
    if provenance["feature_extractor"].get("feature_dim") != features.shape[1]:
        raise ValueError(name + " provenance dimension differs")
    return {"features": features, "groups": groups.astype(np.int64), "ids": ids,
            "provenance": provenance}


def _selected_group(dataset, group, cap, seed):
    """Order-independent, outcome-free ID-hash selection, followed by stable IDs."""
    indexes = np.flatnonzero(dataset["groups"] == group).tolist()
    ranked = sorted(indexes, key=lambda i: (
        hashlib.sha256(f"{seed}|{group}|{dataset['ids'][i]}".encode()).digest(),
        dataset["ids"][i],
    ))
    if cap is not None:
        ranked = ranked[:cap]
    ranked.sort(key=lambda i: dataset["ids"][i])
    if len(ranked) < 2:
        raise ValueError(f"Group{group} has fewer than2 samples")
    return dataset["features"][ranked], dataset["ids"][ranked].tolist()


def evaluate_feature_groups(real, fake, analysis_cap=512, selection_seed=1701, block_size=256):
    """All four requested-bin KIDs with unequal real/fake counts and provenance.

    The fixed cap applies before looking at features. The primary point estimate
    and bootstrap must use the same selected rows; this makes runtime adaptation
    explicit rather than putting a small-subset interval around a larger estimate.
    """
    real, fake = _checked_dataset(real, "real"), _checked_dataset(fake, "fake")
    selection_seed = _positive_int(selection_seed, "selection_seed", 0)
    if analysis_cap is not None:
        analysis_cap = _positive_int(analysis_cap, "analysis_cap", 2)
    if real["provenance"]["feature_extractor"] != fake["provenance"]["feature_extractor"]:
        raise ValueError("Feature extractor/weights/preprocessing provenance differs")
    group_results, values = {}, []
    for group in GROUPS:
        x, xid = _selected_group(real, group, analysis_cap, selection_seed)
        y, yid = _selected_group(fake, group, analysis_cap, selection_seed)
        score = polynomial_kid(x, y, block_size)
        values.append(score)
        group_results[str(group)] = {
            "annotation": GROUP_LABELS[group], "kid": score,
            "real_available": int(np.sum(real["groups"] == group)),
            "fake_requested_available": int(np.sum(fake["groups"] == group)),
            "n_real": len(x), "n_fake": len(y),
            "real_ids": xid, "fake_requested_ids": yid,
        }
    extractor = real["provenance"]["feature_extractor"]
    weight_sha = extractor.get("weights_sha256", "")
    complete = (isinstance(weight_sha, str) and len(weight_sha) == 64
                and all(c in "0123456789abcdef" for c in weight_sha)
                and bool(extractor.get("preprocessing"))
                and bool(extractor.get("implementation_version")))
    return {
        "status": "computed", "metric_version": METRIC_VERSION,
        "computed_utc": datetime.now(timezone.utc).isoformat(),
        "feature_provenance_complete": complete,
        "feature_definition": extractor,
        "feature_definition_sha256": _json_hash(extractor),
        "estimator": "unbiased_cubic_MMD_U_statistic_unequal_counts_float64",
        "kernel": "(dot(x,y)/feature_dim+1)^3",
        "units": "raw_KID_not_multiplied_by1000", "analysis_cap": analysis_cap,
        "selection_seed": int(selection_seed), "group_encoding": "2*Male+Blond_Hair",
        "requested_bins_used": True, "generated_filtering": False,
        "generated_relabeling": False, "groups": group_results,
        "primary_tail_group": 3, "primary_tail_kid": values[3],
        "macro_kid": float(np.mean(values)), "worst_group_kid": float(np.max(values)),
        "worst_group_is_unbiased": False,
        "fid": {"status": "not_computed", "reason": "Optional; not mixed with KID or mislabeled Clean-FID"},
        "real_provenance": real["provenance"], "fake_provenance": fake["provenance"],
        "evaluation_source_sha256": sha256_file(__file__), "numpy_version": np.__version__,
    }


def _bootstrap_scores(x, y, real_counts, fake_counts):
    """Exact U-statistic on bootstrap multisets, excluding positions, not values."""
    kxx, kyy, kxy = _kernel(x, x), _kernel(y, y), _kernel(x, y)
    n, m = len(x), len(y)
    xx = (np.einsum("bi,bi->b", real_counts @ kxx, real_counts)
          - real_counts @ np.diag(kxx)) / (n * (n - 1))
    yy = (np.einsum("bi,bi->b", fake_counts @ kyy, fake_counts)
          - fake_counts @ np.diag(kyy)) / (m * (m - 1))
    xy = np.einsum("bi,bi->b", real_counts @ kxy, fake_counts) / (n * m)
    return xx + yy - 2 * xy


def paired_bootstrap(real, fake_a, fake_b, replicates=2000, seed=9101,
                     analysis_cap=512, selection_seed=1701, confidence=0.95,
                     max_seconds=None, _return_draws=False):
    """Approximate paired image/latent bootstrap conditional on TWO fixed models.

    No training-seed resampling. Identical latent IDs and requested bins are
    required across A/B. Real-image resampling is not an identity-cluster CI.
    max_seconds is a soft budget checked between groups: an ongoing BLAS call
    cannot be interrupted. On overshoot return INCOMPLETE without an interval;
    never select a shorter successful prefix or change the estimator afterward.
    """
    replicates = _positive_int(replicates, "replicates", 2)
    seed = _positive_int(seed, "seed", 0)
    selection_seed = _positive_int(selection_seed, "selection_seed", 0)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in(0,1)")
    if analysis_cap is not None:
        analysis_cap = _positive_int(analysis_cap, "analysis_cap", 2)
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    real = _checked_dataset(real, "real")
    a, b = _checked_dataset(fake_a, "fake_a"), _checked_dataset(fake_b, "fake_b")
    spec = real["provenance"]["feature_extractor"]
    if a["provenance"]["feature_extractor"] != spec or b["provenance"]["feature_extractor"] != spec:
        raise ValueError("Bootstrap feature provenance differs")
    amap = dict(zip(a["ids"].tolist(), a["groups"].tolist()))
    bmap = dict(zip(b["ids"].tolist(), b["groups"].tolist()))
    if amap != bmap:
        raise ValueError("Paired bootstrap requires identical latent IDs and requested groups")
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    score_a, score_b = [], []
    selected = {}
    for group in GROUPS:
        if max_seconds is not None and time.monotonic() - start >= max_seconds:
            return {"status": "INCOMPLETE", "reason": "bootstrap_time_cap",
                    "completed_groups": group, "requested_replicates": replicates,
                    "confidence_interval": None, "conditional_on_fixed_models": True}
        x, xid = _selected_group(real, group, analysis_cap, selection_seed)
        y, yid = _selected_group(a, group, analysis_cap, selection_seed)
        z, zid = _selected_group(b, group, analysis_cap, selection_seed)
        if yid != zid:
            raise ValueError("Selected paired latent IDs differ")
        rc = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=replicates).astype(np.float64)
        fc = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=replicates).astype(np.float64)
        score_a.append(_bootstrap_scores(x, y, rc, fc))
        score_b.append(_bootstrap_scores(x, z, rc, fc))
        selected[str(group)] = {"real_ids": xid, "fake_requested_ids": yid}
    if max_seconds is not None and time.monotonic() - start >= max_seconds:
        return {"status": "INCOMPLETE", "reason": "bootstrap_time_cap",
                "completed_groups": 4, "requested_replicates": replicates,
                "confidence_interval": None, "conditional_on_fixed_models": True}
    aa, bb = np.stack(score_a, axis=1), np.stack(score_b, axis=1)
    tails = (1 - confidence) / 2
    intervals = {}
    difference_draws = {}
    endpoints = {f"group{g}": (aa[:, g], bb[:, g]) for g in GROUPS}
    endpoints.update(tail=(aa[:, 3], bb[:, 3]), macro=(aa.mean(1), bb.mean(1)),
                     worst=(aa.max(1), bb.max(1)))
    for endpoint, (left, right) in endpoints.items():
        diff = left - right
        if _return_draws:
            difference_draws[endpoint] = diff
        intervals[endpoint] = {
            "a_minus_b_percentile_interval": np.quantile(diff, [tails, 1 - tails]).tolist(),
            "a_minus_b_one_sided_upper": float(np.quantile(diff, confidence)),
        }
    report = {
        "status": "computed", "method": "paired_percentile_multinomial_image_and_latent_bootstrap",
        "conditional_on_fixed_models": True, "training_seed_uncertainty_included": False,
        "identity_cluster_uncertainty_included": False, "coverage_guaranteed": False,
        "confidence": confidence, "replicates": replicates, "seed": seed,
        "analysis_cap": analysis_cap, "selection_seed": selection_seed,
        "difference_direction": "A_minus_B_lower_is_better",
        "group_max_recomputed_each_replicate": True,
        "selected_ids": selected, "intervals": intervals,
        "elapsed_seconds": time.monotonic() - start,
        "limitations": ["fixed trained networks", "finite image reference, not population certificate",
                        "image-level resampling; identity clusters not modeled", "percentile coverage approximate"],
    }
    if _return_draws:
        report["_difference_draws"] = difference_draws
    return report


def paired_bootstrap_fixed_models(real, fake_models_a, fake_models_b, **kwargs):
    """Mean endpoint differences over fixed trained models, NOT seed inference.

    Every model must use the same evaluation latent IDs/conditions (common
    random numbers). Resample those IDs and the shared real references jointly
    across all fixed networks. The worst endpoint is mean-of-per-model maxima.
    No model/seed is resampled and no seed-population coverage is asserted.
    """
    if not fake_models_a or len(fake_models_a) != len(fake_models_b):
        raise ValueError("Need nonempty equally sized sequences of fixed model pairs")
    if "_return_draws" in kwargs:
        raise ValueError("_return_draws is internal")
    reference_ids = None
    for dataset in list(fake_models_a) + list(fake_models_b):
        checked = _checked_dataset(dataset, "fixed model")
        ids = dict(zip(checked["ids"].tolist(), checked["groups"].tolist()))
        if reference_ids is None:
            reference_ids = ids
        elif reference_ids != ids:
            raise ValueError("All fixed models need the same common evaluation latent IDs")
    started = time.monotonic()
    max_seconds = kwargs.pop("max_seconds", None)
    all_draws = []
    template = None
    for a, b in zip(fake_models_a, fake_models_b):
        remaining = None if max_seconds is None else max_seconds - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            return {"status": "INCOMPLETE", "reason": "bootstrap_time_cap",
                    "confidence_interval": None, "conditional_on_fixed_models": True}
        report = paired_bootstrap(real, a, b, max_seconds=remaining, _return_draws=True, **kwargs)
        if report["status"] != "computed":
            return report
        all_draws.append(report.pop("_difference_draws"))
        template = report
    confidence = template["confidence"]
    tail = (1 - confidence) / 2
    for endpoint in template["intervals"]:
        draws = np.mean(np.stack([item[endpoint] for item in all_draws]), axis=0)
        template["intervals"][endpoint] = {
            "a_minus_b_percentile_interval": np.quantile(draws, [tail, 1 - tail]).tolist(),
            "a_minus_b_one_sided_upper": float(np.quantile(draws, confidence)),
        }
    template.update(method="joint_common_latent_reference_bootstrap_mean_fixed_model_endpoints",
                    fixed_model_pairs=len(all_draws), model_or_training_seed_resampling=False,
                    worst_endpoint="mean_of_per_model_group_maxima",
                    evaluation_latent_ids_shared_across_fixed_models=True,
                    elapsed_seconds=time.monotonic() - started)
    return template


def extract_feature_batches(extractor, batches, provenance=None):
    """batches yield (uint8_BHWC_RGB, string_ids, requested_or_real_groups)."""
    all_features, all_groups, all_ids = [], [], []
    for images, ids, groups in batches:
        images = rgb_uint8(images)
        features = _matrix(extractor(images), "extracted features", min_rows=1)
        if len(features) != len(images) or len(ids) != len(images) or len(groups) != len(images):
            raise ValueError("Extractor changed the batch count")
        all_features.append(features)
        all_groups.extend(groups)
        all_ids.extend(ids)
    if not all_features:
        raise ValueError("No image batches")
    metadata = dict(provenance or {})
    metadata["feature_extractor"] = dict(extractor.provenance)
    return _checked_dataset({"features": np.concatenate(all_features),
                             "groups": np.asarray(all_groups), "ids": np.asarray(all_ids),
                             "provenance": metadata}, "extracted")


class TorchvisionInceptionFeatures:
    """Optional 2048-dimensional pooled torchvision ImageNet Inception features.

    Uses official weights/transforms, including resize342 then center-crop299.
    This is not the TensorFlow-FID/Clean-FID Inception definition. The actual
    complete downloaded/local weights file SHA256 is always recorded.
    """
    def __init__(self, device="cpu", weights_path=None, expected_weights_sha256=None,
                 allow_download=False):
        import inspect
        import torch
        import torchvision
        from torchvision.models import inception_v3, Inception_V3_Weights

        weights = Inception_V3_Weights.IMAGENET1K_V1
        cache_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
        path = Path(weights_path) if weights_path is not None else cache_path
        if not path.is_file():
            if not allow_download or weights_path is not None:
                raise FileNotFoundError("Feature weights absent; download only with explicit allow_download=True")
            weights.get_state_dict(progress=True, check_hash=True)
        actual_hash = sha256_file(path)
        official_prefix = Path(urlparse(weights.url).path).stem.rsplit("-", 1)[-1]
        if not actual_hash.startswith(official_prefix):
            raise ValueError("Feature weights do not match the official URL hash prefix")
        if expected_weights_sha256 is not None and actual_hash != expected_weights_sha256:
            raise ValueError("Feature weights SHA256 mismatch")
        # Mirror the documented pretrained builder without random initialization.
        model = inception_v3(weights=None, aux_logits=True, transform_input=True, init_weights=False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.preprocess = weights.transforms(antialias=True)
        self.device, self.torch = device, torch
        source_path = inspect.getsourcefile(inspect.unwrap(inception_v3))
        preprocess_source = inspect.getsourcefile(type(self.preprocess))
        self.provenance = {
            "feature_id": "torchvision_inception_v3_IMAGENET1K_V1_pooled_2048",
            "standard_cleanfid": False, "feature_dim": 2048,
            "implementation_version": torchvision.__version__, "torch_version": torch.__version__,
            "weights_enum": "Inception_V3_Weights.IMAGENET1K_V1",
            "weights_url": weights.url, "weights_sha256": actual_hash,
            "weights_file_name": path.name,
            "implementation_source_sha256": sha256_file(source_path) if source_path else None,
            "preprocess_source_sha256": sha256_file(preprocess_source) if preprocess_source else None,
            "preprocessing": {
                "input": "uint8_BHWC_RGB_0_255",
                "generated_quantization": "numpy_rint((tanh_output+1)*127.5); no clipping",
                "preset": repr(self.preprocess),
                "resize": "bilinear342_antialias_true_then_center_crop299",
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "model_transform_input": True,
            },
        }

    def __call__(self, images):
        images = rgb_uint8(images)
        tensor = self.torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
        with self.torch.inference_mode():
            value = self.model(self.preprocess(tensor).to(self.device))
        return value.detach().cpu().numpy().astype(np.float64)


def generate_feature_dataset(generator, extractor, seed, count_per_group=2000,
                             latent_dim=128, batch_size=64, device="cpu",
                             provenance=None, sample_sink=None):
    """Root-owned generator(z,long_labels) -> BCHW RGB[-1,1], always all requests.

    The parent selects/loads the EMA checkpoint. A sink may save predetermined
    grids, but receives every unfiltered batch and cannot change metric rows.
    """
    import torch

    was_training = generator.training
    generator.eval()

    def batches():
        for item in generation_inputs(seed, count_per_group, latent_dim, batch_size):
            with torch.inference_mode():
                images = generator(torch.from_numpy(item["latents"]).to(device),
                                   torch.from_numpy(item["groups"]).to(device))
            if images.ndim != 4 or images.shape[1] != 3 or tuple(images.shape[2:]) != (64, 64):
                raise ValueError("Generator must produce BCHW RGB64x64")
            pixels = rgb_uint8(images.detach().float().cpu().permute(0, 2, 3, 1).numpy())
            if len(pixels) != len(item["ids"]):
                raise ValueError("Generator dropped requests")
            if sample_sink is not None:
                sample_sink(pixels.copy(), item["ids"].copy(), item["groups"].copy())
            yield pixels, item["ids"], item["groups"]

    metadata = dict(provenance or {})
    metadata.update(generation_seed=int(seed), generation_count_per_group=int(count_per_group),
                    latent_dim=int(latent_dim), latent_rng="numpy_PCG64_SeedSequence_seed_group_80317",
                    filtering=False, relabeling=False)
    try:
        return extract_feature_batches(extractor, batches(), metadata)
    finally:
        generator.train(was_training)


def save_feature_dataset(dataset, path):
    dataset = _checked_dataset(dataset, "save")
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Feature path must end.npz")
    sidecar = path.with_suffix(".json")
    if path.exists() or path.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("Refusing to overwrite feature data/provenance")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, features=dataset["features"], groups=dataset["groups"], ids=dataset["ids"])
    metadata = {"provenance": dataset["provenance"], "npz_sha256": sha256_file(path),
                "n_samples": len(dataset["features"]), "feature_dim": dataset["features"].shape[1]}
    with sidecar.open("x") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return metadata


def load_feature_dataset(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if sha256_file(path) != metadata["npz_sha256"]:
        raise ValueError("Feature archive SHA256 differs from provenance")
    with np.load(path, allow_pickle=False) as archive:
        dataset = {key: archive[key] for key in ("features", "groups", "ids")}
    dataset["provenance"] = metadata["provenance"]
    dataset = _checked_dataset(dataset, str(path))
    if len(dataset["features"]) != metadata["n_samples"] or dataset["features"].shape[1] != metadata["feature_dim"]:
        raise ValueError("Feature metadata shape mismatch")
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, type=Path, help="real.npz with hash-bound real.json")
    parser.add_argument("--fake", required=True, type=Path, help="fake.npz with requested groups")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--analysis-cap", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=1701)
    parser.add_argument("--paired-baseline", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.975)
    parser.add_argument("--bootstrap-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refusing to overwrite evaluation output")
    real, fake = load_feature_dataset(args.real), load_feature_dataset(args.fake)
    answer = evaluate_feature_groups(real, fake, args.analysis_cap, args.selection_seed)
    if args.paired_baseline:
        answer["paired_bootstrap"] = paired_bootstrap(
            real, fake, load_feature_dataset(args.paired_baseline),
            replicates=args.bootstrap_replicates, confidence=args.bootstrap_confidence,
            analysis_cap=args.analysis_cap, selection_seed=args.selection_seed,
            max_seconds=args.bootstrap_seconds,
        )
    roles = {"real": args.real, "fake": args.fake}
    if args.paired_baseline:
        roles["paired_baseline"] = args.paired_baseline
    answer["input_files"] = {
        role: {"name": path.name, "sha256": sha256_file(path),
               "provenance_sha256": sha256_file(path.with_suffix(".json"))}
        for role, path in roles.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(answer, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": answer["status"], "output": str(args.output),
                      "primary_tail_kid": answer["primary_tail_kid"],
                      "feature_provenance_complete": answer["feature_provenance_complete"]}))


if __name__ == "__main__":
    main()
