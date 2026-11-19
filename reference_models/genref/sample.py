"""Sample a reference checkpoint; no metric or benchmark is computed."""
import argparse
import json
from pathlib import Path
import torch
from .data import requested_labels, sha256
from .registry import create


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--class-label", type=int)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--sampler", choices=["ancestral", "langevin", "sde", "ode",
                                             "euler", "heun", "consistency"])
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    sidecar = Path(str(args.output) + ".json")
    if args.output.exists() or args.output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("Choose new sample and sidecar paths")
    if args.count <= 0 or args.threads <= 0:
        parser.error("Positive count and threads required")
    torch.set_num_threads(args.threads)
    if (not args.checkpoint.is_file() or args.checkpoint.is_symlink()
            or args.checkpoint.stat().st_size > 1024 ** 3):
        raise ValueError("Expected a regular reference checkpoint within the 1GiB cap")
    before = sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "genref_v1":
        raise ValueError("Unsupported reference checkpoint")
    source = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    if checkpoint.get("source_sha256") != source:
        raise ValueError("Reference source differs from checkpoint provenance")
    model = create(checkpoint["model"], checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    labels = requested_labels(args.count, model.classes, args.class_label, args.device)
    # Reset after construction so requested sample noise is independent of initialization draws.
    torch.manual_seed(args.seed)
    with torch.no_grad():
        samples = model.sample(args.count, labels, args.sample_steps, args.sampler)
    if samples.shape != (args.count, model.dim) or not bool(torch.isfinite(samples).all()):
        raise FloatingPointError("Nonfinite or incorrectly shaped samples")
    if sha256(args.checkpoint) != before:
        raise RuntimeError("Checkpoint changed during sampling")
    payload = {"samples": samples.cpu().reshape(args.count, *checkpoint["observation_shape"]),
               "labels": None if labels is None else labels.cpu()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    result = {"status": "REFERENCE_SAMPLES_WRITTEN", "model": checkpoint["model"],
              "benchmark_evaluation": False, "count": args.count, "seed": args.seed,
              "sample_steps": args.sample_steps, "sampler": args.sampler,
              "checkpoint_sha256": before, "samples_sha256": sha256(args.output),
              "source_sha256_matches_checkpoint": True,
              "output_shape": list(payload["samples"].shape),
              "samples_clipped": False, "torch": torch.__version__}
    sidecar.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

