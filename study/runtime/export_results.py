"""Private-user export of source-bound trained artifacts, never raw CelebA.

Only explicit project paths are read. No network, browser, credentials, Drive,
training or model inference is used. Torch/NumPy are lazy. --self-test uses toy
files only. Main-cohort export never discovers or silently merges P extensions.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


CHUNK_LIMIT = 512 * 1024
RAW_CHUNK_BYTES = (CHUNK_LIMIT // 77) * 57  #76 base64 chars+LF per57 raw bytes
FIXED_ZIP_DATE = (2026, 11, 18, 0, 0, 0)
NOTICE = """PRIVATE RESEARCH ARTIFACTS; NO BLANKET PUBLIC LICENSE GRANT

These archives contain newly authored research code, records, generated samples
and/or trained EMA generator tensors. No raw CelebA images or official raw
annotation/partition files are included. No pretrained Inception/ResNet weights,
judge checkpoint, full discriminator/optimizer checkpoint, browser state,
authentication data or secrets are included.

CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
The dataset authors restrict the dataset to non-commercial research and restrict
redistribution. This private artifact delivery is not a new grant of dataset,
image, identity, publicity, model-weight or commercial-use rights. Generated
images and trained weights are not certified free of memorization or privacy
risk. The image generator's conditions use literal dataset annotation names.

Third-party imports retain their own terms. PyTorch/torchvision have BSD-style
licenses; clean-fid and the author PRDC implementation use MIT licenses. Our
adapters import clean-fid; pretrained feature/classifier weights are NOT bundled.
Known projection/SN/cBN, gSR, UTLO, KID/PRDC and quantification antecedents remain
credited in source/report records. Do not construe their licenses as a blanket
MIT license for this entire research package or its data/model artifacts.

This package records a finite-budget experiment. It grants no semantic-fairness,
state-of-the-art, universal superiority, privacy or human-label correctness
certificate. Both judge adequacy gates failed in the main experiment; any narrow
image-feature quality result does not override that measurement limitation.
"""

INFERENCE_EXAMPLE = r'''"""EMA-only inference; no dataset, feature weights or judge needed."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import Generator

p=argparse.ArgumentParser()
p.add_argument("--checkpoint",required=True)
p.add_argument("--output",required=True)
p.add_argument("--device",default="cpu")
p.add_argument("--seed",type=int,default=78231)
p.add_argument("--per-group",type=int,default=16)
a=p.parse_args()
if a.per_group<1: raise ValueError("per-group must be positive")
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
record=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
g=Generator(**record["generator_config"])
g.load_state_dict(record["ema_generator"],strict=True)
g.eval().requires_grad_(False).to(a.device)
for group in range(4):
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([a.seed,group,80317])))
    z=rng.standard_normal((a.per_group,record["generator_config"]["z_dim"]),dtype=np.float32)
    with torch.inference_mode():
        x=g(torch.from_numpy(z).to(a.device),torch.full((a.per_group,),group,dtype=torch.long,device=a.device))
    x=x.float().cpu().permute(0,2,3,1).numpy()
    if not np.isfinite(x).all() or x.min()<-1 or x.max()>1: raise ValueError("Invalid generator output")
    pixels=np.rint((x.astype(np.float64)+1)*127.5).astype(np.uint8)
    for i,image in enumerate(pixels): Image.fromarray(image).save(out/f"group{group}_{i:04d}.png")
print("Generated all requested groups without filtering; this is not a semantic-label verification.")
'''

FEATURE_REPLAY = r'''"""CPU replay on exact selected feature inputs; no GPU or model weights."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code"))
from evaluate import load_feature_dataset,evaluate_feature_groups,paired_bootstrap_fixed_models
p=argparse.ArgumentParser()
p.add_argument("--bootstrap",action="store_true",help="Slower: reproduce all four fixed-model interval attempts")
p.add_argument("--output",default="replayed_metrics.json")
a=p.parse_args()
expected=json.loads((ROOT/"expected_points.json").read_text())
real=load_feature_dataset(ROOT/"features/real_test.npz")
fakes={};reports={}
for key,item in expected["models"].items():
    fake=load_feature_dataset(ROOT/item["feature_file"]);fakes[key]=fake
    result=evaluate_feature_groups(real,fake,analysis_cap=512,selection_seed=1701)
    for group in map(str,range(4)):
        if result["groups"][group]["real_ids"]!=item["kid"]["groups"][group]["real_ids"]:raise ValueError("Real IDs differ")
        if result["groups"][group]["fake_requested_ids"]!=item["kid"]["groups"][group]["fake_requested_ids"]:raise ValueError("Fake IDs differ")
        if not np.isclose(result["groups"][group]["kid"],item["kid"]["groups"][group]["kid"],rtol=1e-10,atol=1e-10):raise ValueError("KID differs")
    reports[key]=result
answer={"point_KID_checks":"passed","models":reports,"bootstrap":{}}
if a.bootstrap:
    for candidate,reference in [("V1","B"),("V2","B"),("V1","GSR"),("V2","GSR")]:
        name=candidate+"_vs_"+reference
        original=expected.get("bootstraps",{}).get(name)
        if not original:continue
        archived=json.loads((ROOT/original).read_text())
        if archived.get("status")!="computed":
            answer["bootstrap"][name]={"status":"original_attempt_not_computed_no_replacement","original_status":archived.get("status")}
            continue
        keys=[f"{arm}_seed{s}" for arm in [candidate,reference] for s in [101,202,303]]
        if not all(k in fakes for k in keys):continue
        replayed=paired_bootstrap_fixed_models(real,
            [fakes[f"{candidate}_seed{s}"] for s in [101,202,303]],
            [fakes[f"{reference}_seed{s}"] for s in [101,202,303]],
            replicates=2000,seed=4703,analysis_cap=512,selection_seed=1701,confidence=.9875)
        if replayed.get("status")!="computed":
            answer["bootstrap"][name]={"status":"replay_not_computed_original_unchanged","replay_status":replayed.get("status")}
            continue
        for endpoint in ["tail","macro","worst"]:
            if not np.isclose(replayed["intervals"][endpoint]["a_minus_b_one_sided_upper"],archived["intervals"][endpoint]["a_minus_b_one_sided_upper"],rtol=1e-10,atol=1e-10):raise ValueError("Bootstrap differs")
        answer["bootstrap"][name]=replayed
with Path(a.output).open("x") as f:json.dump(answer,f,indent=2,allow_nan=False)
print("CPU replay finished; fixed-network bootstrap coverage is approximate, not training-seed inference.")
'''


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
        tmp.replace(path)
    else:
        with path.open("x") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")


def inside(root, value):
    path = Path(value)
    if path.is_absolute():
        try:
            path = root / path.relative_to("/content/generative_longtail")
        except ValueError:
            pass
    else:
        require(".." not in path.parts, "Unsafe relative input")
        path = root / path
    require(not path.is_symlink(), "Symlink input refused")
    resolved = path.resolve()
    require(resolved.is_relative_to(root.resolve()), "Input outside the explicit study root refused")
    return resolved


def safe_name(name):
    value = PurePosixPath(name)
    require(not value.is_absolute() and ".." not in value.parts and "\\" not in name,
            "Unsafe archive member")
    return value.as_posix()


def file_record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Missing/symlink input: " + str(path))
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def make_archive(destination, files, metadata):
    """Stream allowlisted payloads; verify every ZIP entry against its manifest."""
    destination = Path(destination)
    require(not destination.exists() and not destination.is_symlink(), "Refusing archive overwrite")
    require(len(files) == len(set(files)), "Duplicate payload names")
    payload = {safe_name(name): {"path": str(path), **file_record(path)} for name, path in sorted(files.items())}
    require("MANIFEST.json" not in payload, "Manifest name reserved")
    manifest = {"metadata": metadata, "manifest_excludes_itself": True,
                "files": {name: {k: value[k] for k in ("bytes", "sha256")} for name, value in payload.items()}}
    encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = destination.with_name(destination.name + ".partial")
    require(not temporary.exists() and not temporary.is_symlink(), "Partial archive preserved; review before retry")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for name, item in payload.items():
            info = zipfile.ZipInfo(name, FIXED_ZIP_DATE); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with Path(item["path"]).open("rb") as src, z.open(info, "w", force_zip64=True) as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
        z.writestr("MANIFEST.json", encoded)
    verify_archive(temporary)
    temporary.replace(destination)
    return {**file_record(destination), "archive": destination.name,
            "manifest_sha256": hashlib.sha256(encoded).hexdigest(), "payload_files": len(payload)}


def verify_archive(path, expected=None):
    if expected is not None:
        require(file_record(path) == {k: expected[k] for k in ("bytes", "sha256")}, "Archive hash/size changed")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP members")
        for name in names:
            safe_name(name)
        data = z.read("MANIFEST.json")
        manifest = json.loads(data)
        require(set(names) == set(manifest["files"]) | {"MANIFEST.json"}, "ZIP manifest membership differs")
        for name, item in manifest["files"].items():
            h, count = hashlib.sha256(), 0
            with z.open(name) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk); count += len(chunk)
            require(count == item["bytes"] and h.hexdigest() == item["sha256"], "ZIP payload changed: " + name)
        if expected is not None:
            require(hashlib.sha256(data).hexdigest() == expected["manifest_sha256"], "Internal manifest changed")
    return manifest


def armour_archive(archive, output):
    """Independent base64 chunks, at most512KiB ASCII each,76-column LF lines."""
    archive, output = Path(archive), Path(output)
    original = file_record(archive)
    prefix = archive.stem + "--" + original["sha256"][:16]
    folder = output / (prefix + ".b64chunks")
    require(not folder.is_symlink(), "Symlink chunk folder refused before reading its manifest")
    if folder.exists():
        manifest = read(folder / "CHUNKS.json")
        require(manifest["archive_sha256"] == original["sha256"] and manifest["archive_bytes"] == original["bytes"], "Existing chunk folder differs")
        require(manifest["chunk_count"] == len(manifest["chunks"])
                and len({item["filename"] for item in manifest["chunks"]}) == len(manifest["chunks"]), "Invalid reused chunk list")
        # Validate ALL names before the first file_record/read, not afterward.
        for item in manifest["chunks"]:
            name = item["filename"]
            require(Path(name).name == name and name not in (".", "..") and "\\" not in name,
                    "Unsafe reused chunk filename")
        for item in manifest["chunks"]:
            require(file_record(folder / item["filename"]) == {k: item[k] for k in ("bytes", "sha256")}, "Existing chunk changed")
        digest, count = hashlib.sha256(), 0
        for raw in verified_chunk_bytes(folder / "CHUNKS.json"):
            digest.update(raw); count += len(raw)
        require(count == original["bytes"] and digest.hexdigest() == original["sha256"], "Existing chunks do not reconstruct the archive")
        return folder / "CHUNKS.json"
    folder.mkdir(parents=True, exist_ok=False)
    chunks = []
    with archive.open("rb") as stream:
        for number, raw in enumerate(iter(lambda: stream.read(RAW_CHUNK_BYTES), b""), 1):
            encoded = base64.b64encode(raw)
            text = b"\n".join(encoded[i:i + 76] for i in range(0, len(encoded), 76)) + b"\n"
            require(len(text) <= CHUNK_LIMIT and text.isascii(), "Chunk size/ASCII invariant failed")
            name = f"{prefix}--chunk{number:05d}.b64.txt"
            path = folder / name
            with path.open("xb") as f: f.write(text)
            chunks.append({"filename": name, **file_record(path), "decoded_bytes": len(raw),
                           "decoded_sha256": hashlib.sha256(raw).hexdigest()})
    require(file_record(archive) == original, "Archive changed during armour")
    manifest = {"archive_name": archive.name, "archive_sha256": original["sha256"],
                "archive_bytes": original["bytes"], "chunk_count": len(chunks),
                "max_encoded_bytes_per_chunk": CHUNK_LIMIT, "columns": 76,
                "encoding": "base64_ASCII_LF_final_newline", "chunks": chunks}
    write(folder / "CHUNKS.json", manifest)
    return folder / "CHUNKS.json"


def verified_chunk_bytes(manifest_path):
    """Bounded-memory verified decoder iterator; only CRLF→LF is normalized."""
    manifest_path = Path(manifest_path)
    manifest = read(manifest_path)
    require(manifest["chunk_count"] == len(manifest["chunks"]), "Chunk count differs")
    require(len({c["filename"] for c in manifest["chunks"]}) == len(manifest["chunks"]), "Duplicate chunk filenames")
    for item in manifest["chunks"]:
        require(Path(item["filename"]).name == item["filename"] and item["filename"] not in (".", ".."), "Unsafe chunk filename")
        path = manifest_path.parent / item["filename"]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= CHUNK_LIMIT + CHUNK_LIMIT // 76 + 2,
                "Missing/symlink/oversized copied chunk")
        text = path.read_bytes().replace(b"\r\n", b"\n")
        require(len(text) <= CHUNK_LIMIT and len(text) == item["bytes"]
                and hashlib.sha256(text).hexdigest() == item["sha256"], "Copied chunk hash differs; recopy it")
        require(text.isascii() and text.endswith(b"\n") and all(len(line) <= 76 for line in text.splitlines()), "Invalid armour format")
        raw = base64.b64decode(text.replace(b"\n", b""), validate=True)
        require(len(raw) == item["decoded_bytes"] and hashlib.sha256(raw).hexdigest() == item["decoded_sha256"], "Decoded chunk differs")
        yield raw


def decode_chunks(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = read(manifest_path)
    if output.exists():
        require(file_record(output) == {"bytes": manifest["archive_bytes"], "sha256": manifest["archive_sha256"]}, "Existing decoded output differs")
        return file_record(output)
    require(not output.is_symlink(), "Symlink decode output refused")
    partial = output.with_name(output.name + ".partial")
    require(not partial.exists() and not partial.is_symlink(), "Partial decoded file preserved; inspect before retry")
    output.parent.mkdir(parents=True, exist_ok=True)
    h, count = hashlib.sha256(), 0
    with partial.open("xb") as stream:
        for raw in verified_chunk_bytes(manifest_path):
            stream.write(raw); h.update(raw); count += len(raw)
    require(count == manifest["archive_bytes"] and h.hexdigest() == manifest["archive_sha256"], "Decoded archive hash differs")
    partial.replace(output)
    return file_record(output)


def load_context(root, evaluation, analysis, allow_incomplete):
    protocol_path = root / "protocol_frozen.json"
    protocol = read(protocol_path)
    require(protocol["status"] == "frozen_before_main_training_and_comparative_quality", "Need actual frozen protocol")
    binding_path, registry_path = evaluation / "extra_evaluation_bindings.json", evaluation / "artifact_sha256.json"
    binding, registry = read(binding_path), read(registry_path)
    require(binding["status"] == "frozen_before_first_evaluation_quality_output", "Need actual pre-quality evaluation binding")
    require(binding["protocol_sha256"] == sha(protocol_path), "Evaluation protocol binding differs")
    require(registry.get("extra_evaluation_bindings.json") == sha(binding_path), "Evaluation binding not registered")
    require(sha(root / "evaluation_sources_approved.json") == binding["source_approval_sha256"], "Source approval record changed")
    require(sha(root / "main_config.json") == protocol["main_config_file_sha256"], "Main config changed")
    for filename, field in (("judge/validation_report.json", "judge_validation_sha256"), ("judge_test/test_report.json", "judge_test_sha256")):
        require(sha(root / filename) == binding[field], "Bound judge report changed")
    for source_map in (protocol["source_sha256"], binding["source_sha256"]):
        for name, digest in source_map.items():
            require(Path(name).name == name and sha(root / "src" / name) == digest, "Frozen source changed: " + name)
    for name, digest in registry.items():
        safe_name(name)
        require(sha(inside(evaluation, name)) == digest, "Registered evaluation artifact changed: " + name)
    result_path = analysis / "analysis.json"
    result = read(result_path) if result_path.exists() else None
    require(result is not None or allow_incomplete, "Analysis missing; --allow-incomplete required for explicitly partial delivery")
    if result:
        require(result["source_sha256"] == binding["source_sha256"]["analyze_results.py"], "Analyzer source differs")
        require(result["input_files"]["protocol"]["sha256"] == sha(protocol_path), "Analysis protocol differs")
        for name, item in result["input_files"].items():
            path = item.get("path") if name == "protocol" else name
            require(sha(inside(root, path)) == item["sha256"], "Analyzed input changed: " + str(path))
    planned = [(int(seed), str(arm)) for seed, arm in protocol["matrix"]["order"]]
    require(len(planned) == len(set(planned)) == 12, "Main cohort must contain exactly12 unique planned runs")
    require(set(protocol["matrix"]["arms"]) == {"B", "V1", "V2", "GSR"}
            and protocol["matrix"]["training_seeds"] == [101, 202, 303], "This exporter is only the frozen main cohort")
    completed, missing = {}, []
    for seed, arm in planned:
        key = f"{arm}_seed{seed}"
        path = root / "runs" / key / "complete.json"
        if not path.exists():
            missing.append({"arm": arm, "seed": seed, "reason": "missing_completion"}); continue
        record = read(path)
        if record.get("status") != "complete":
            missing.append({"arm": arm, "seed": seed, "reason": "not_complete"}); continue
        require(record["arm"] == arm and record["seed"] == seed and record["completed_steps"] == 20000, "Wrong completed-run identity")
        expected = binding["checkpoints"].get(key)
        require(expected and sha(path) == expected["complete_sha256"] and record["checkpoint_sha256"] == expected["sha256"], "Run differs from pre-quality binding")
        require(sha(root / "runs" / key / "latest.pt") == expected["sha256"], "Full checkpoint changed/missing")
        completed[key] = record
    require(not missing or allow_incomplete, "Incomplete main cohort; explicit --allow-incomplete required")
    require(completed, "No completed trained model; refusing an empty generator archive")
    return {"root": root, "evaluation": evaluation, "analysis": analysis, "protocol": protocol,
            "binding": binding, "registry": registry, "analysis_record": result, "completed": completed,
            "missing": missing, "protocol_sha256": sha(protocol_path), "binding_sha256": sha(binding_path),
            "registry_sha256": sha(registry_path), "analysis_sha256": sha(result_path) if result else None}


def model_payload(context, stage):
    import torch
    root, records = context["root"], {}
    stage.mkdir(parents=True, exist_ok=False)
    for key, record in context["completed"].items():
        source = root / "runs" / key / "latest.pt"
        # A trusted, locally trained full checkpoint; SHA checked before pickle.
        require(sha(source) == record["checkpoint_sha256"], "Full checkpoint changed before EMA extraction")
        full = torch.load(source, map_location="cpu", weights_only=False)
        require(full["arm"] == record["arm"] and full["seed"] == record["seed"] and full["step"] == 20000, "Checkpoint identity differs")
        require(full["config"] == context["protocol"]["training_config"], "Checkpoint training config differs")
        state = {}
        for name, value in full["ema_generator"].items():
            require(isinstance(name, str) and isinstance(value, torch.Tensor), "EMA contains a non-tensor state entry")
            require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "Nonfinite EMA tensor")
            state[name] = value.detach().cpu().contiguous()
        config = json.loads(json.dumps(full["config"], allow_nan=False))
        payload = {"kind": "EMA_generator_only", "ema_generator": state,
            "generator_config": {"z_dim": config["latent_dim"], "num_classes": 4, "base_channels": config["base_channels"]},
            "config": config, "arm": record["arm"], "seed": record["seed"], "step": 20000,
            "step_definition": "attempted_D_G_iterations",
            "successful_G_updates": record["successful_g_updates"],
            "successful_D_updates": record["successful_d_updates"],
            "original_full_checkpoint_sha256": record["checkpoint_sha256"],
            "models_py_sha256": context["protocol"]["source_sha256"]["models.py"],
            "protocol_sha256": context["protocol_sha256"]}
        target = stage / "weights" / (key + "_EMA.pt"); target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
        checked = torch.load(target, map_location="cpu", weights_only=True)
        require(set(checked["ema_generator"]) == set(state), "EMA roundtrip keys differ")
        require(all(torch.equal(checked["ema_generator"][name], value) for name, value in state.items()), "EMA tensor roundtrip differs")
        records[key] = {"arm": record["arm"], "seed": record["seed"], "step": 20000,
                        "file": "weights/" + key + "_EMA.pt", "tensor_entries": len(state),
                        "original_full_checkpoint_sha256": record["checkpoint_sha256"], **file_record(target)}
        del full, payload, checked, state
    shutil.copyfile(root / "src" / "models.py", stage / "models.py")
    (stage / "generate.py").write_text(INFERENCE_EXAMPLE)
    (stage / "requirements-inference.txt").write_text("torch>=2.6\nnumpy>=1.24\nPillow>=9.1\n")
    (stage / "LICENSE_NOTICE.txt").write_text(NOTICE)
    write(stage / "MODEL_RECORDS.json", records)
    return {p.relative_to(stage).as_posix(): p for p in stage.rglob("*") if p.is_file()}


def research_payload(context):
    root, evaluation, analysis = context["root"], context["evaluation"], context["analysis"]
    files = {}
    def add(path, name):
        if path.is_file():
            resolved = inside(root, path)
            files[safe_name(name)] = resolved
    # Every current main-study source is explicitly protocol/evaluation-bound.
    # Future extension/auth/helper files are not silently swept into this cohort.
    for name in sorted(set(context["protocol"]["source_sha256"]) | set(context["binding"]["source_sha256"])):
        add(root / "src" / name, "code/src/" + name)
    for name in ("test_metrics.py", "test_diversity.py", "test_analyze_results.py", "test_calibration_sensitivity.py"):
        add(root / "tests" / name, "code/tests/" + name)
    for name in ("protocol_frozen.json", "main_config.json", "evaluation_sources_approved.json",
                 "matrix_summary.json", "matrix_execution.jsonl", "pipeline_smoke.json", "feature_provenance.json"):
        add(root / name, "records/" + name)
    for seed, arm in context["protocol"]["matrix"]["order"]:
        key = f"{arm}_seed{seed}"
        for name in ("start.json", "complete.json", "training.jsonl", "exposures.npy", "failure.json"):
            add(root / "runs" / key / name, "records/training/" + key + "/" + name)
        for path in (root / "failures").glob(key + "_*.json"):
            add(path, "records/training_failures/" + path.name)
    for arm in ("B", "V1", "V2", "GSR"):
        for folder in (root / "pilot").glob(arm + "_seed*"):
            for name in ("start.json", "complete.json", "training.jsonl", "failure.json"):
                add(folder / name, "records/no_quality_throughput_pilots/" + folder.name + "/" + name)
    for name in ("training_config.json", "training.jsonl", "calibration.json", "validation_report.json",
                 "freeze.json", "freeze_candidate.json", "complete.json", "attempt_started.json", "failure.json"):
        add(root / "judge" / name, "records/judge/" + name)
    add(root / "judge_test" / "test_report.json", "records/judge_test/test_report.json")
    for name in ("status.json", "results_index.json", "artifact_sha256.json", "extra_evaluation_bindings.json", "final_integrity.json"):
        add(evaluation / name, "records/evaluation/" + name)
    for pattern in ("attempt*.stdout.log", "attempt*.stderr.log", "attempt*.process.json"):
        for path in evaluation.glob(pattern): add(path, "records/evaluation/" + path.name)
    for name in context["registry"]:
        parts = PurePosixPath(name).parts
        allowed = (parts[0] in {"metrics", "bootstrap", "nearest_training", "secondary_sensitivity", "judge_predictions"}
                   and Path(name).suffix in {".json", ".npz"})
        allowed = allowed or (parts[0] == "generation" and (name.endswith(".png") or parts[-1] in {"generation_complete.json", "pixel_duplicates.json"}))
        allowed = allowed or (parts[0] == "cache_diagnostics" and parts[-1] in {"summary.json", "duplicate_groups.json"})
        allowed = allowed or (parts[0] == "references" and parts[-1] == "features.json")
        if allowed: add(evaluation / name, "records/evaluation/" + name)
    for path in (evaluation / "failures").glob("*.json"):
        add(path, "records/evaluation/failures/" + path.name)
    if analysis.exists():
        names = ["analysis.json", "seed_results.csv", "seed_pair_differences.csv"]
        names.extend("annotation_bootstrap_" + a + "_vs_" + b + ".json"
                     for a, b in context["protocol"]["evaluation"]["bootstrap"]["comparisons"])
        for name in names:
            add(analysis / name, "records/analysis/" + name)
    # Explicitly authored helper source may be launched from outside the study
    # root; this exception never discovers/reads arbitrary external data paths.
    files["code/export_results.py"] = Path(__file__).resolve()
    return files


def lossless_float32(values):
    import numpy as np
    values = np.asarray(values)
    require(values.dtype.kind == "f", "Feature values must be floating point")
    compact = values.astype(np.float32)
    require(np.isfinite(compact).all() and np.array_equal(compact.astype(values.dtype), values),
            "Float32 conversion is NOT lossless; refusing a rounded feature export")
    return compact


def selected_feature_payload(context, stage):
    import numpy as np
    root, evaluation = context["root"], context["evaluation"]
    stage.mkdir(parents=True, exist_ok=False)
    expected = {"models": {}, "scope": "exact_point_selected_inputs_not_full_dataset", "counts": {},
                "bootstraps": {}, "missing_or_uncommitted_point_models": []}
    real_ids = None
    def selected(source, destination, group_ids, expected_sha):
        require(sha(source) == expected_sha, "Feature source differs from point metric")
        metadata = read(source.with_suffix(".json"))
        require(metadata["npz_sha256"] == expected_sha, "Feature provenance differs")
        with np.load(source, allow_pickle=False) as data:
            ids, groups, features = data["ids"].astype(str), data["groups"], data["features"]
        require(features.ndim == 2 and features.dtype.kind == "f" and len(features) == len(ids)
                and groups.shape == ids.shape and features.shape[1] == 2048, "Invalid full feature schema")
        lookup = {name: i for i, name in enumerate(ids)}
        require(len(lookup) == len(ids), "Feature IDs are not unique")
        ordered = [name for g in range(4) for name in group_ids[str(g)]]
        require(len(set(ordered)) == len(ordered), "Point IDs overlap groups")
        indexes = [lookup[name] for name in ordered]
        chosen = features[indexes]
        compact = lossless_float32(chosen)
        actual_groups = groups[indexes]
        require(actual_groups.tolist() == [g for g in range(4) for _ in group_ids[str(g)]], "Point group IDs differ")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as f:
            np.savez_compressed(f, features=compact, groups=actual_groups, ids=np.asarray(ordered))
        provenance = dict(metadata["provenance"])
        provenance["export_selection"] = "exact_saved_point_KID_IDs; numeric_lossless_float32"
        write(destination.with_suffix(".json"), {"provenance": provenance, "npz_sha256": sha(destination),
              "n_samples": len(ordered), "feature_dim": compact.shape[1], "original_full_feature_npz_sha256": expected_sha})
        return len(ordered)
    for key in context["completed"]:
        path = evaluation / "metrics" / (key + ".json")
        registry_name = "metrics/" + key + ".json"
        if registry_name not in context["registry"]:
            expected["missing_or_uncommitted_point_models"].append({"model": key, "reason": "point_metric_not_committed"})
            continue
        require(path.is_file() and sha(path) == context["registry"][registry_name], "Committed point metric changed")
        metric = read(path)
        require(metric["checkpoint"]["sha256"] == context["completed"][key]["checkpoint_sha256"]
                == context["binding"]["checkpoints"][key]["sha256"]
                and metric["checkpoint"]["step"] == 20000
                and metric["arm"] == context["completed"][key]["arm"]
                and metric["seed"] == context["completed"][key]["seed"],
                "Point metric belongs to a different completed checkpoint")
        require(metric["protocol_sha256"] == context["protocol_sha256"]
                and metric["kid"]["feature_definition"] == context["protocol"]["feature_provenance"],
                "Point metric protocol/feature definition differs")
        kid = metric["kid"]
        require(kid["status"] == "computed", "Invalid point KID")
        ids = {str(g): kid["groups"][str(g)]["real_ids"] for g in range(4)}
        if real_ids is None:
            real_ids = ids
            expected["counts"]["real"] = selected(evaluation / "references/real_test/features.npz", stage / "features/real_test.npz", ids, metric["real_feature_npz_sha256"])
        else:
            require(real_ids == ids, "Real point subsets differ across models")
        fake_ids = {str(g): kid["groups"][str(g)]["fake_requested_ids"] for g in range(4)}
        filename = "features/" + key + ".npz"
        expected["counts"][key] = selected(evaluation / "generation" / key / "features.npz", stage / filename,
                                           fake_ids, metric["fake_feature_npz_sha256"])
        expected["models"][key] = {"feature_file": filename, "kid": kid,
                                   "generator_checkpoint_sha256": metric["checkpoint"]["sha256"]}
    require(real_ids is not None and expected["models"], "No point-selected features available")
    for candidate, reference in context["protocol"]["evaluation"]["bootstrap"]["comparisons"]:
        name = candidate + "_vs_" + reference
        path = evaluation / "bootstrap" / (name + ".json")
        if path.exists():
            require(context["registry"].get("bootstrap/" + name + ".json") == sha(path), "Unregistered bootstrap report")
            target = stage / "expected_bootstraps" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            expected["bootstraps"][name] = target.relative_to(stage).as_posix()
    write(stage / "expected_points.json", expected)
    (stage / "reproduce_selected_metrics.py").write_text(FEATURE_REPLAY)
    (stage / "LICENSE_NOTICE.txt").write_text(NOTICE)
    (stage / "requirements-CPU.txt").write_text("numpy>=1.24\n")
    (stage / "code").mkdir()
    shutil.copyfile(root / "src/evaluate.py", stage / "code/evaluate.py")
    return {p.relative_to(stage).as_posix(): p for p in stage.rglob("*") if p.is_file()}


def build(args):
    root = args.root.resolve()
    evaluation = (args.evaluation or root / "evaluation").resolve()
    analysis = (args.analysis or root / "analysis").resolve()
    output_candidate = args.output or root / "exports"
    require(not output_candidate.is_symlink(), "Symlink exports directory refused")
    output = output_candidate.resolve()
    require(evaluation.is_relative_to(root) and analysis.is_relative_to(root) and output.is_relative_to(root),
            "Build inputs/output must stay inside the explicit private study root")
    context = load_context(root, evaluation, analysis, args.allow_incomplete)
    point_keys = [key for key in context["completed"] if "metrics/" + key + ".json" in context["registry"]]
    bootstrap_status = {}
    for candidate, reference in context["protocol"]["evaluation"]["bootstrap"]["comparisons"]:
        key = candidate + "_vs_" + reference
        name = "bootstrap/" + key + ".json"
        bootstrap_status[key] = read(evaluation / name).get("status") if name in context["registry"] else "MISSING"
    evaluation_incomplete = len(point_keys) != 12 or any(s != "computed" for s in bootstrap_status.values()) or context["analysis_record"] is None
    require(not evaluation_incomplete or args.allow_incomplete, "Evaluation incomplete; explicit --allow-incomplete required for partial delivery")
    plan = {"exporter_sha256": sha(__file__), "cohort": "main_frozen_12_only_no_implicit_extensions",
            "protocol_sha256": context["protocol_sha256"], "evaluation_binding_sha256": context["binding_sha256"],
            "artifact_registry_sha256": context["registry_sha256"], "analysis_sha256": context["analysis_sha256"],
            "selected_features_requested": args.selected_features, "armour_requested": not args.no_armour,
            "planned_models": 12, "completed_models": len(context["completed"]), "incomplete_models": context["missing"],
            "quality_complete_runs": context["analysis_record"].get("complete_training_and_quality_runs") if context["analysis_record"] else None,
            "committed_point_metric_models": len(point_keys), "primary_bootstrap_status": bootstrap_status,
            "evaluation_incomplete": evaluation_incomplete,
            "nearest_training_records": sum("nearest_training/" + key + ".json" in context["registry"] for key in context["completed"]),
            "secondary_sensitivity_records": sum("secondary_sensitivity/" + key + ".json" in context["registry"] for key in context["completed"]),
            "cache_duplicate_summary_present": "cache_diagnostics/summary.json" in context["registry"]}
    state_path = output / "export_state.json"
    if output.exists():
        require(args.resume, "Exports already exist; only immutable verified --resume is allowed")
        state = read(state_path)
        require(state["plan"] == plan, "Export resume inputs/options changed")
        for name, item in state["archives"].items():
            verify_archive(output / name, item)
            if "armour_manifest" in item:
                require(sha(output / item["armour_manifest"]) == item["armour_manifest_sha256"], "Immutable chunk manifest changed")
    else:
        require(not args.resume, "No existing export state to resume")
        output.mkdir(parents=True, exist_ok=False)
        state = {"status": "export_in_progress", "plan": plan, "created_utc": datetime.now(timezone.utc).isoformat(), "archives": {}}
        write(state_path, state)
    staging = output / ".staging"
    require(not staging.is_symlink(), "Symlink export staging directory refused")
    staging.mkdir(exist_ok=True)
    shared_metadata = {**plan, "license_and_scope": "private_delivery_no_blanket_public_license", "raw_real_images_included": False,
                       "pretrained_or_judge_weights_included": False, "full_optimizer_checkpoints_preserved_in_cloud": True,
                       "source_file_roles": {
                           "main_frozen": sorted(context["protocol"]["source_sha256"]),
                           "prequality_evaluation_frozen": sorted(set(context["binding"]["source_sha256"]) - set(context["protocol"]["source_sha256"])),
                           "unbound_src_files_omitted_not_main_cohort": sorted(p.name for p in (root / "src").glob("*.py") if p.name not in context["binding"]["source_sha256"]),
                       }}
    for name in ["trained_generators.zip", "research_bundle.zip"] + (["selected_features.zip"] if args.selected_features else []):
        if name not in state["archives"]:
            stage = staging / Path(name).stem
            require(not stage.exists() and not stage.is_symlink(), "Partial export stage preserved; inspect before retry: " + str(stage))
            if name == "trained_generators.zip":
                files = model_payload(context, stage)
            elif name == "selected_features.zip":
                files = selected_feature_payload(context, stage)
            else:
                stage.mkdir()
                (stage / "LICENSE_NOTICE.txt").write_text(NOTICE)
                write(stage / "COHORT_COUNTS.json", plan)
                files = research_payload(context)
                files.update({"LICENSE_NOTICE.txt": stage / "LICENSE_NOTICE.txt", "COHORT_COUNTS.json": stage / "COHORT_COUNTS.json"})
            state["archives"][name] = make_archive(output / name, files, {**shared_metadata, "archive_role": Path(name).stem})
            write(state_path, state, replace=True)
        if not args.no_armour:
            manifest = armour_archive(output / name, output)
            state["archives"][name]["armour_manifest"] = manifest.relative_to(output).as_posix()
            state["archives"][name]["armour_manifest_sha256"] = sha(manifest)
            write(state_path, state, replace=True)
    # Inputs are revalidated rather than claiming preserved originals by intent.
    check = load_context(root, evaluation, analysis, args.allow_incomplete)
    require(check["protocol_sha256"] == plan["protocol_sha256"] and check["binding_sha256"] == plan["evaluation_binding_sha256"]
            and check["registry_sha256"] == plan["artifact_registry_sha256"] and check["analysis_sha256"] == plan["analysis_sha256"],
            "Source records changed during export")
    state["status"] = ("export_completed_incomplete_training_cohort" if context["missing"] else
                       "export_completed_partial_evaluation" if evaluation_incomplete else "export_completed")
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write(state_path, state, replace=True)
    if not (output / "export_manifest.json").exists(): write(output / "export_manifest.json", state)
    else: require(read(output / "export_manifest.json")["archives"] == state["archives"], "Existing final export manifest differs")
    print(json.dumps(state, indent=2))


def self_test():
    """Toy file/ZIP/armour only; no project records, Torch, GPU or real data."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "toy.bin").write_bytes(bytes(range(256)) * 4000)
        (root / "note.txt").write_text("toy fixture only\n")
        archive = root / "toy.zip"
        record = make_archive(archive, {"toy.bin": root / "toy.bin", "note.txt": root / "note.txt"}, {"synthetic_fixture": True})
        verify_archive(archive, record)
        # Incompressible-enough raw file forces multiple transport chunks.
        raw = root / "raw.bin"
        with raw.open("wb") as f:
            for i in range(40000): f.write(hashlib.sha256(str(i).encode()).digest())
        manifest = armour_archive(raw, root)
        require(read(manifest)["chunk_count"] > 1, "Multichunk fixture did not span chunks")
        require(armour_archive(raw, root) == manifest, "Immutable verified armour reuse failed")
        decoded = root / "decoded.bin"
        decode_chunks(manifest, decoded)
        require(file_record(raw) == file_record(decoded), "Chunk roundtrip failed")
        decode_chunks(manifest, decoded)  # immutable verified reuse
        first = manifest.parent / read(manifest)["chunks"][0]["filename"]
        original = first.read_bytes()
        first.write_bytes(original.replace(b"\n", b"\r\n"))
        decode_chunks(manifest, root / "CRLF_decoded.bin")
        require(file_record(raw) == file_record(root / "CRLF_decoded.bin"), "CRLF transport normalization failed")
        first.write_bytes(original)
        original = first.read_bytes(); first.write_bytes((b"A" if original[:1] != b"A" else b"B") + original[1:])
        try: decode_chunks(manifest, root / "corrupted.bin")
        except RuntimeError: pass
        else: raise RuntimeError("Chunk corruption not detected")
        try: make_archive(archive, {"note": root / "note.txt"}, {})
        except RuntimeError: pass
        else: raise RuntimeError("Archive overwrite not refused")
        try: safe_name("../escape")
        except RuntimeError: pass
        else: raise RuntimeError("Unsafe member path not refused")
        import numpy as np
        exact = np.asarray([np.float64(np.float32(.1)), 0.0, -1.0])
        require(np.array_equal(lossless_float32(exact).astype(np.float64), exact), "Lossless feature fixture")
        try: lossless_float32(np.asarray([np.pi]))
        except RuntimeError: pass
        else: raise RuntimeError("Lossy feature conversion not refused")
    print("PASS: toy ZIP, payload hashes, multichunk armour, roundtrip, corruption, no-overwrite and path fixtures; no study data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--root", type=Path, default=Path("/content/generative_longtail"))
    p.add_argument("--evaluation", type=Path)
    p.add_argument("--analysis", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--selected-features", action="store_true")
    p.add_argument("--no-armour", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("armour"); p.add_argument("--archive", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("decode"); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "build": build(args)
    elif args.command == "armour": print(armour_archive(args.archive, args.output))
    elif args.command == "decode": print(json.dumps(decode_chunks(args.manifest, args.output)))
    else: self_test()


if __name__ == "__main__":
    main()
