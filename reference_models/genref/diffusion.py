"""Finite Gaussian DDPM with epsilon prediction and ancestral sampling.

Primary: arXiv:2006.11239. The tiny schedule is pedagogical, not a quality preset.
"""
import torch
from .networks import TimeMLP, VectorModel


class DDPM(VectorModel):
    def __init__(self, dim, width=64, classes=0, diffusion_steps=32,
                 beta_start=1e-4, beta_end=.2, **unused):
        super().__init__(dim, classes)
        if diffusion_steps < 2 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("Invalid DDPM schedule")
        self.timesteps = int(diffusion_steps)
        self.denoiser = TimeMLP(dim, width, classes)
        beta = torch.linspace(beta_start, beta_end, self.timesteps)
        alpha = 1 - beta
        abar = alpha.cumprod(0)
        previous = torch.cat([torch.ones(1), abar[:-1]])
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("abar", abar)
        self.register_buffer("posterior_variance", beta * (1 - previous) / (1 - abar))

    def predict_noise(self, x, t, labels=None):
        return self.denoiser(x, t.to(x.dtype) / (self.timesteps - 1), labels)

    def loss(self, x, labels=None):
        self.check_x(x, labels)
        t = torch.randint(self.timesteps, (len(x),), device=x.device)
        noise = torch.randn_like(x)
        abar = self.abar[t, None]
        noisy = abar.sqrt() * x + (1 - abar).sqrt() * noise
        value = (self.predict_noise(noisy, t, labels) - noise).square().mean()
        return value, {"epsilon_mse": value.detach()}

    @torch.no_grad()
    def sample(self, n, labels=None, steps=None, sampler=None):
        if steps not in (None, self.timesteps) or sampler not in (None, "ancestral"):
            raise ValueError("DDPM uses every configured ancestral step; no hidden step skipping")
        labels = self.sample_labels(n, labels)
        x = torch.randn(n, self.dim, device=self.beta.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=x.device, dtype=torch.long)
            noise = self.predict_noise(x, t, labels)
            mean = (x - self.beta[i] * noise / (1 - self.abar[i]).sqrt()) / self.alpha[i].sqrt()
            x = mean if i == 0 else mean + self.posterior_variance[i].sqrt() * torch.randn_like(x)
        return x

