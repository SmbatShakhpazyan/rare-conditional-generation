"""Create tiny SYNTHETIC local tensors for functionality checks, not research data."""
import argparse
from pathlib import Path
import torch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Demo output must be new")
    if args.dim < 1 or args.classes < 0 or args.count < max(2, args.classes):
        parser.error("Invalid dimensions / class coverage")
    torch.manual_seed(args.seed)
    labels = torch.arange(args.count) % args.classes if args.classes else None
    phase = torch.zeros(args.count) if labels is None else labels.float() * (6.283185307 / args.classes)
    coordinates = torch.linspace(0, 6.283185307, args.dim)
    x = (.6 * torch.sin(coordinates[None, :] + phase[:, None]) + .15 * torch.randn(args.count, args.dim)).tanh()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "labels": labels}, args.output)
    print("SYNTHETIC functionality tensor saved; not an empirical benchmark dataset")


if __name__ == "__main__":
    main()

