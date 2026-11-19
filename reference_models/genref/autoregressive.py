"""Strictly causal masked Gaussian autoregression.

Reference mechanism: NADE, proceedings.mlr.press/v15/larochelle11a.html.
This masked MLP is an original Gaussian adaptation, not the paper's binary NADE.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .networks import VectorModel, labels_for


class MaskedLinear(nn.Linear):
    def __init__(self, d_in, d_out, mask):
        super().__init__(d_in, d_out)
        self.register_buffer("mask", mask.float())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class AutoregressiveGaussian(VectorModel):
    def __init__(self, dim, width=64, classes=0, **unused):
        super().__init__(dim, classes)
        inputs = torch.arange(1, dim + 1)
        # Degree-zero units see no observed coordinate, but do see labels/bias.
        # They let the first conditional depend on the requested class while
        # every output still excludes its own coordinate and all later ones.
        hidden = torch.arange(width) % dim
        outputs = inputs.repeat(2)
        self.first = MaskedLinear(dim, width, hidden[:, None] >= inputs[None, :])
        self.middle = MaskedLinear(width, width, hidden[:, None] >= hidden[None, :])
        self.last = MaskedLinear(width, 2 * dim, outputs[:, None] > hidden[None, :])
        self.embedding = nn.Embedding(classes, width) if classes else None

    def parameters_of_conditionals(self, x, labels=None):
        labels = self.check_x(x, labels)
        h = self.first(x)
        if self.embedding is not None:
            h = h + self.embedding(labels)
        h = F.silu(h)
        h = F.silu(self.middle(h))
        mean, logstd = self.last(h).chunk(2, -1)
        return mean, logstd.clamp(-5, 2)

    def log_prob(self, x, labels=None):
        mean, logstd = self.parameters_of_conditionals(x, labels)
        return (-.5 * ((x - mean) * (-logstd).exp()).square()
                - logstd - .5 * math.log(2 * math.pi)).sum(-1)

    def loss(self, x, labels=None):
        nll = -self.log_prob(x, labels).mean()
        return nll, {"negative_log_likelihood": nll.detach()}

    @torch.no_grad()
    def sample(self, n, labels=None, steps=None, sampler=None):
        self.no_sampler_options(steps, sampler)
        labels = self.sample_labels(n, labels)
        x = torch.zeros(n, self.dim, device=next(self.parameters()).device)
        for i in range(self.dim):
            mean, logstd = self.parameters_of_conditionals(x, labels)
            x[:, i] = mean[:, i] + logstd[:, i].exp() * torch.randn_like(mean[:, i])
        return x

