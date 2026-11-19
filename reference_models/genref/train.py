"""Bounded reference training on a user-supplied local TRAIN tensor.

Run from this library directory: python -m genref.train --help
No validation selection, benchmark claim, external dataset or implicit resume.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import time
import torch
from .data import draw_batch, load_training_tensor, requested_labels, sha256
from .registry import Config, MODELS, create, preset


def fresh_directory(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Choose a new output directory; no overwrite or implicit resume")
    path.mkdir(parents=True)
    return path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def optimizer_step(loss, optimizer, parameters):
    if loss.ndim or not bool(torch.isfinite(loss)):
        raise FloatingPointError("Loss is not a finite scalar")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("Missing or nonfinite gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ae-steps", type=int, default=25,
                        help="Additional autoencoder updates before latent diffusion")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=300.)
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.threads) <= 0 or args.ae_steps < 0:
        parser.error("Positive steps/batch/threads required; ae-steps must be nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Positive finite learning rate required")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("Positive finite time cap required")
    if args.model == "latent_diffusion" and args.ae_steps < 1:
        parser.error("Latent diffusion requires an explicit positive autoencoder stage")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    output = fresh_directory(args.output)
    record = {"status": "RUNNING_REFERENCE_ONLY", "model": args.model,
              "benchmark_experiment": False, "python": platform.python_version(),
              "torch": torch.__version__, "device": args.device, "seed": args.seed,
              "max_seconds": args.max_seconds, "steps_completed": 0,
              "autoencoder_steps_completed": 0}
    write_json(output / "run.json", record)
    def deadline():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("Soft time cap reached between bounded operations")
    try:
        x, labels, shape, classes, input_hash = load_training_tensor(args.data, args.unconditional)
        config = preset(args.preset, x.shape[1], classes)
        model = create(args.model, config).to(args.device).train()
        record.update(config=asdict(config), observation_shape=shape,
                      input_sha256=input_hash, observations=len(x),
                      sampling="balanced_with_replacement" if args.balanced else "natural_with_replacement",
                      source_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
        write_json(output / "run.json", record)
        log = open(output / "losses.jsonl", "x", buffering=1)
        def batch():
            a, b = draw_batch(x, labels, args.batch_size, args.balanced)
            return a.to(args.device), None if b is None else b.to(args.device)
        def emit(phase, step, loss, metrics, gradnorm):
            row = {"phase": phase, "step": step, "loss": float(loss.detach()),
                   "gradient_norm_before_clip": gradnorm,
                   "elapsed_seconds": time.monotonic() - started,
                   **{k: float(v) for k, v in metrics.items()}}
            log.write(json.dumps(row, allow_nan=False) + "\n")
        try:
            if args.model == "latent_diffusion":
                params = list(model.autoencoder.parameters())
                ae_optimizer = torch.optim.Adam(params, lr=args.learning_rate)
                for step in range(1, args.ae_steps + 1):
                    deadline()
                    a, b = batch()
                    loss, metrics = model.autoencoder.loss(a, b)
                    norm = optimizer_step(loss, ae_optimizer, params)
                    emit("autoencoder", step, loss, metrics, norm)
                    record["autoencoder_steps_completed"] = step
                # Calibration is a separate full pass over this TRAIN tensor.
                deadline()
                model.freeze_and_calibrate(x.to(args.device),
                                           None if labels is None else labels.to(args.device))
                params = list(model.diffusion.parameters())
                optimizer = torch.optim.Adam(params, lr=args.learning_rate)
            elif args.model == "cgan":
                dparams = list(model.discriminator.parameters())
                gparams = list(model.generator.parameters())
                d_optimizer = torch.optim.Adam(dparams, lr=args.learning_rate, betas=(0., .9))
                g_optimizer = torch.optim.Adam(gparams, lr=args.learning_rate, betas=(0., .9))
            else:
                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(params, lr=args.learning_rate)
            for step in range(1, args.steps + 1):
                deadline()
                a, b = batch()
                if args.model == "cgan":
                    model.discriminator.train().requires_grad_(True)
                    dloss, dmetrics = model.discriminator_loss(a, b)
                    dnorm = optimizer_step(dloss, d_optimizer, dparams)
                    emit("discriminator", step, dloss, dmetrics, dnorm)
                    # Freeze SN state as well as critic parameters during G update.
                    model.discriminator.eval().requires_grad_(False)
                    loss, metrics = model.generator_loss(len(a), b)
                    norm = optimizer_step(loss, g_optimizer, gparams)
                    model.discriminator.train().requires_grad_(True)
                else:
                    loss, metrics = model.loss(a, b)
                    norm = optimizer_step(loss, optimizer, params)
                    if hasattr(model, "after_step"):
                        model.after_step()
                record["steps_completed"] = step
                emit("model", step, loss, metrics, norm)
                write_json(output / "run.json", record)
        finally:
            log.close()
        deadline()
        if sha256(args.data) != input_hash:
            raise RuntimeError("Training input changed")
        checkpoint = {"format": "genref_v1", "model": args.model,
                      "config": asdict(config), "observation_shape": shape,
                      "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "input_sha256": input_hash, "source_sha256": record["source_sha256"],
                      "steps_completed": args.steps,
                      "autoencoder_steps_completed": record["autoencoder_steps_completed"]}
        torch.save(checkpoint, output / "checkpoint.pt")
        record.update(status="REFERENCE_TRAINING_COMPLETE", elapsed_seconds=time.monotonic() - started,
                      checkpoint_sha256=sha256(output / "checkpoint.pt"),
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
        write_json(output / "run.json", record)
        print(json.dumps(record, allow_nan=False))
    except BaseException as error:
        record.update(status="FAILED_OR_INTERRUPTED_PARTIALS_PRESERVED",
                      error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.monotonic() - started)
        write_json(output / "run.json", record)
        raise


if __name__ == "__main__":
    main()

