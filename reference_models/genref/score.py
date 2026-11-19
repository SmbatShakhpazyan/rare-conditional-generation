"""Noise-conditional score matching, annealed Langevin, and VE reverse SDE/ODE.

Primaries: arXiv:1907.05600 and arXiv:2011.13456. Scalar isotropic VE noise;
no predictor-corrector, likelihood integration or exact-score claim.
"""
import math
import torch
from .networks import TimeMLP, VectorModel


class NoiseScore(VectorModel):
    def __init__(self, dim, width=64, classes=0, sigma_min=.01, sigma_max=2.,
                 noise_levels=8, score_step_size=2e-5, **unused):
        super().__init__(dim, classes)
        if not 0 < sigma_min < sigma_max or noise_levels < 2 or score_step_size <= 0:
            raise ValueError("Invalid score noise schedule")
        self.net = TimeMLP(dim, width, classes)
        self.register_buffer("sigmas", torch.logspace(math.log10(sigma_max),
                                                       math.log10(sigma_min), noise_levels))
        self.step_size = float(score_step_size)

    def score(self, x, sigma, labels=None):
        return self.net(x, sigma.log(), labels) / sigma[:, None]

    def loss(self, x, labels=None):
        self.check_x(x, labels)
        index = torch.randint(len(self.sigmas), (len(x),), device=x.device)
        sigma = self.sigmas[index]
        noise = torch.randn_like(x)
        # sigma^2 weighted DSM, expressed without a numerically huge target.
        residual = self.score(x + sigma[:, None] * noise, sigma, labels) * sigma[:, None] + noise
        value = .5 * residual.square().mean()
        return value, {"weighted_denoising_score_mse": value.detach()}

    @torch.no_grad()
    def sample(self, n, labels=None, steps=None, sampler=None):
        if sampler not in (None, "langevin"):
            raise ValueError("NoiseScore supports only annealed Langevin sampling")
        correctors = 5 if steps is None else int(steps)
        if correctors < 1:
            raise ValueError("Positive iterations per noise level required")
        labels = self.sample_labels(n, labels)
        x = torch.randn(n, self.dim, device=self.sigmas.device) * self.sigmas[0]
        for sigma in self.sigmas:
            eta = self.step_size * (sigma / self.sigmas[-1]).square()
            levels = sigma.expand(n)
            for _ in range(correctors):
                x = x + .5 * eta * self.score(x, levels, labels) + eta.sqrt() * torch.randn_like(x)
        return x


class VESDE(VectorModel):
    def __init__(self, dim, width=64, classes=0, sigma_min=.01, sigma_max=2., **unused):
        super().__init__(dim, classes)
        if not 0 < sigma_min < sigma_max:
            raise ValueError("Invalid VE schedule")
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.log_ratio = math.log(sigma_max / sigma_min)
        self.net = TimeMLP(dim, width, classes)

    def sigma(self, time):
        return self.sigma_min * torch.exp(time * self.log_ratio)

    def score(self, x, time, labels=None):
        return self.net(x, time, labels) / self.sigma(time)[:, None]

    def loss(self, x, labels=None):
        self.check_x(x, labels)
        time = torch.rand(len(x), device=x.device) * (1 - .001) + .001
        sigma = self.sigma(time)
        noise = torch.randn_like(x)
        residual = self.score(x + sigma[:, None] * noise, time, labels) * sigma[:, None] + noise
        value = .5 * residual.square().mean()
        return value, {"continuous_weighted_dsm": value.detach()}

    @torch.no_grad()
    def sample(self, n, labels=None, steps=None, sampler=None):
        count = 64 if steps is None else int(steps)
        sampler = "sde" if sampler is None else sampler
        if count < 1 or sampler not in ("sde", "ode"):
            raise ValueError("Choose positive steps and sampler sde or ode")
        labels = self.sample_labels(n, labels)
        device = next(self.parameters()).device
        x = torch.randn(n, self.dim, device=device) * self.sigma_max
        times = torch.linspace(1, .001, count + 1, device=device)
        for current, following in zip(times[:-1], times[1:]):
            # h=-dt>0: reverse SDE drift is +g^2 score in this convention.
            h = current - following
            time = current.expand(n)
            g2 = 2 * self.log_ratio * self.sigma(time).square()[:, None]
            x = x + (1. if sampler == "sde" else .5) * h * g2 * self.score(x, time, labels)
            if sampler == "sde":
                x = x + (g2 * h).sqrt() * torch.randn_like(x)
        return x

