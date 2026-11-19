"""Conditional affine coupling flow. Primary: arXiv:1605.08803."""
import math
import torch
from torch import nn
from .networks import ConditionalMLP, VectorModel


class AffineCoupling(nn.Module):
    def __init__(self, dim, width, mask, classes):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.net = ConditionalMLP(dim, 2 * dim, width, classes)
        # Identity initially; this does not remove subsequent gradients.
        nn.init.zeros_(self.net.net[-1].weight)
        nn.init.zeros_(self.net.net[-1].bias)

    def affine(self, x, labels):
        raw_scale, shift = self.net(x * self.mask, labels).chunk(2, -1)
        scale = 1.5 * raw_scale.tanh() * (1 - self.mask)
        return scale, shift * (1 - self.mask)

    def forward(self, x, labels=None, inverse=False):
        scale, shift = self.affine(x, labels)
        if inverse:
            out = x * self.mask + (x * scale.exp() + shift) * (1 - self.mask)
            return out, scale.sum(-1)
        out = x * self.mask + ((x - shift) * (-scale).exp()) * (1 - self.mask)
        return out, -scale.sum(-1)


class RealNVP(VectorModel):
    def __init__(self, dim, width=64, classes=0, coupling_layers=4, **unused):
        super().__init__(dim, classes)
        if coupling_layers < 2:
            raise ValueError("At least two alternating couplings required")
        self.layers = nn.ModuleList([
            AffineCoupling(dim, width, (torch.arange(dim) + i) % 2, classes)
            for i in range(coupling_layers)])

    def transform(self, x, labels=None, inverse=False):
        self.check_x(x, labels)
        logdet = x.new_zeros(len(x))
        for layer in (reversed(self.layers) if inverse else self.layers):
            x, change = layer(x, labels, inverse=inverse)
            logdet = logdet + change
        return x, logdet

    def log_prob(self, x, labels=None):
        z, logdet = self.transform(x, labels)
        return -.5 * (z.square() + math.log(2 * math.pi)).sum(-1) + logdet

    def loss(self, x, labels=None):
        nll = -self.log_prob(x, labels).mean()
        return nll, {"negative_log_likelihood": nll.detach()}

    @torch.no_grad()
    def sample(self, n, labels=None, steps=None, sampler=None):
        self.no_sampler_options(steps, sampler)
        labels = self.sample_labels(n, labels)
        z = torch.randn(n, self.dim, device=next(self.parameters()).device)
        return self.transform(z, labels, inverse=True)[0]

