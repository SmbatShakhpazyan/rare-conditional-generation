"""Small vector networks shared by the reference algorithms.

Images are flattened by the CLI. These are deliberately MLP references, not
copies of any paper's image backbone. No batch normalization is used, so each
sample has a well-defined per-example conditional function.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def labels_for(labels, n: int, classes: int, device):
    if classes == 0:
        if labels is not None:
            raise ValueError("Unconditional model does not accept labels")
        return None
    if labels is None:
        raise ValueError("Conditional model requires one label per sample")
    if labels.dtype != torch.long or tuple(labels.shape) != (n,):
        raise ValueError("Labels must be a one-dimensional int64 tensor")
    labels = labels.to(device)
    if bool(((labels < 0) | (labels >= classes)).any()):
        raise ValueError("Label outside configured class range")
    return labels


def mlp(d_in: int, d_out: int, width: int, depth: int = 2):
    if min(d_in, d_out, width, depth) <= 0:
        raise ValueError("Positive network dimensions required")
    layers = [nn.Linear(d_in, width), nn.SiLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.SiLU()]
    layers.append(nn.Linear(width, d_out))
    return nn.Sequential(*layers)


class ConditionalMLP(nn.Module):
    def __init__(self, d_in, d_out, width, classes=0, depth=2):
        super().__init__()
        self.classes = int(classes)
        self.condition_dim = min(32, width) if classes else 0
        self.embedding = nn.Embedding(classes, self.condition_dim) if classes else None
        self.net = mlp(d_in + self.condition_dim, d_out, width, depth)

    def forward(self, x, labels=None):
        labels = labels_for(labels, len(x), self.classes, x.device)
        if self.embedding is not None:
            x = torch.cat([x, self.embedding(labels)], dim=-1)
        return self.net(x)


class TimeMLP(nn.Module):
    """Sinusoidal scalar-time/noise embedding plus a conditional vector MLP."""
    def __init__(self, dim, width=64, classes=0, time_dim=32):
        super().__init__()
        if time_dim < 4 or time_dim % 2:
            raise ValueError("time_dim must be even and >=4")
        freq = torch.exp(torch.linspace(0, math.log(1000), time_dim // 2))
        self.register_buffer("frequencies", freq)
        self.net = ConditionalMLP(dim + time_dim, dim, width, classes, depth=3)

    def forward(self, x, time, labels=None):
        if time.ndim == 0:
            time = time.expand(len(x))
        if tuple(time.shape) != (len(x),):
            raise ValueError("Time must have one scalar per observation")
        phase = time[:, None] * self.frequencies[None, :]
        emb = torch.cat([phase.sin(), phase.cos()], dim=-1)
        return self.net(torch.cat([x, emb.to(x.dtype)], dim=-1), labels)


class VectorModel(nn.Module):
    def __init__(self, dim, classes=0):
        super().__init__()
        if dim <= 0 or classes < 0:
            raise ValueError("Invalid dimension or class count")
        self.dim, self.classes = int(dim), int(classes)

    def check_x(self, x, labels=None):
        if x.ndim != 2 or x.shape[1] != self.dim or not x.is_floating_point():
            raise ValueError("Expected a floating N-by-dim matrix")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Nonfinite observations")
        return labels_for(labels, len(x), self.classes, x.device)

    def sample_labels(self, n, labels):
        if n <= 0:
            raise ValueError("Positive sample count required")
        return labels_for(labels, n, self.classes, next(self.parameters()).device)

    @staticmethod
    def no_sampler_options(steps, sampler):
        if steps is not None or sampler is not None:
            raise ValueError("This model has no iterative sampler options")

