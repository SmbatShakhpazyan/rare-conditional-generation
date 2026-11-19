"""Standalone adjacent-noise consistency training with an EMA target.

Primary: proceedings.mlr.press/v202/song23a.html. Same-noise adjacent pairs;
fixed geometric grid, L2 distance, tiny MLP: pedagogical deviations from paper.
This is not consistency distillation and requires no pretrained score teacher.
"""
import copy
import math
import torch
from .networks import TimeMLP, VectorModel


class Consistency(VectorModel):
    def __init__(self, dim, width=64, classes=0, sigma_min=.002, sigma_max=20.,
                 noise_levels=16, consistency_ema=.95, **unused):
        super().__init__(dim, classes)
        if not 0 < sigma_min < sigma_max or noise_levels < 2 or not 0 <= consistency_ema < 1:
            raise ValueError("Invalid consistency schedule")
        self.net = TimeMLP(dim, width, classes)
        self.target = copy.deepcopy(self.net).requires_grad_(False)
        self.target.eval()
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.ema_decay = float(consistency_ema)
        self.register_buffer("sigmas", torch.logspace(math.log10(sigma_min),
                                                       math.log10(sigma_max), noise_levels))

    def consistency(self, x, sigma, labels=None, target=False):
        delta = sigma[:, None] - self.sigma_min
        data_variance = .25
        skip = data_variance / (delta.square() + data_variance)
        out = .5 * delta / (sigma[:, None].square() + data_variance).sqrt()
        net = self.target if target else self.net
        return skip * x + out * net(x, sigma.log(), labels)

    def loss(self, x, labels=None):
        self.check_x(x, labels)
        index = torch.randint(1, len(self.sigmas), (len(x),), device=x.device)
        low, high = self.sigmas[index - 1], self.sigmas[index]
        noise = torch.randn_like(x)
        self.target.eval()
        with torch.no_grad():
            goal = self.consistency(x + low[:, None] * noise, low, labels, target=True)
        predicted = self.consistency(x + high[:, None] * noise, high, labels)
        value = (predicted - goal).square().mean()
        return value, {"adjacent_noise_consistency_mse": value.detach()}

    @torch.no_grad()
    def after_step(self):
        for target, online in zip(self.target.parameters(), self.net.parameters()):
            target.lerp_(online, 1 - self.ema_decay)
        for target, online in zip(self.target.buffers(), self.net.buffers()):
            target.copy_(online)

    @torch.no_grad()
    def sample(self, n, labels=None, steps=None, sampler=None):
        count = 1 if steps is None else int(steps)
        if count < 1 or sampler not in (None, "consistency"):
            raise ValueError("Choose positive refinement count and consistency sampler")
        labels = self.sample_labels(n, labels)
        device = next(self.parameters()).device
        times = torch.logspace(math.log10(self.sigma_max), math.log10(self.sigma_min),
                               count + 1, device=device)[:-1]
        x = torch.randn(n, self.dim, device=device) * self.sigma_max
        x = self.consistency(x, times[0].expand(n), labels, target=True)
        for sigma in times[1:]:
            x = x + (sigma.square() - self.sigma_min ** 2).clamp_min(0).sqrt() * torch.randn_like(x)
            x = self.consistency(x, sigma.expand(n), labels, target=True)
        return x

