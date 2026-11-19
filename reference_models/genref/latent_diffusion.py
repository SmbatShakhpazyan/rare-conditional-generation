"""Two-stage Gaussian-VAE compression plus latent DDPM.

Primary inspiration: arXiv:2112.10752. This vector hybrid has no perceptual
encoder, image adversary, cross-attention or pretrained external components.
"""
import torch
from .networks import VectorModel
from .vae import VAE
from .diffusion import DDPM


class LatentDiffusion(VectorModel):
    def __init__(self, dim, width=64, latent_dim=8, classes=0, **kwargs):
        super().__init__(dim, classes)
        self.autoencoder = VAE(dim, width, latent_dim, classes)
        self.diffusion = DDPM(latent_dim, width=width, classes=classes, **kwargs)
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))
        self.register_buffer("statistics_ready", torch.tensor(False))

    @torch.no_grad()
    def freeze_and_calibrate(self, x, labels=None, batch_size=256):
        """One TRAIN-tensor pass; posterior means are the chosen compressed data."""
        self.check_x(x, labels)
        self.autoencoder.eval().requires_grad_(False)
        total = torch.zeros_like(self.latent_mean, dtype=torch.float64)
        squares = torch.zeros_like(total)
        n = 0
        for start in range(0, len(x), batch_size):
            y = None if labels is None else labels[start:start + batch_size]
            mean, _ = self.autoencoder.encode(x[start:start + batch_size], y)
            mean = mean.double()
            total += mean.sum(0)
            squares += mean.square().sum(0)
            n += len(mean)
        if n < 2:
            raise ValueError("At least two training observations required for latent calibration")
        center = total / n
        scale = (squares / n - center.square()).clamp_min(1e-6).sqrt()
        self.latent_mean.copy_(center)
        self.latent_std.copy_(scale)
        self.statistics_ready.fill_(True)

    def loss(self, x, labels=None):
        self.check_x(x, labels)
        if not bool(self.statistics_ready):
            raise RuntimeError("Train the autoencoder, then freeze_and_calibrate before latent diffusion")
        self.autoencoder.eval().requires_grad_(False)
        with torch.no_grad():
            mean, _ = self.autoencoder.encode(x, labels)
            z = (mean - self.latent_mean) / self.latent_std
        return self.diffusion.loss(z, labels)

    @torch.no_grad()
    def sample(self, n, labels=None, steps=None, sampler=None):
        if not bool(self.statistics_ready):
            raise RuntimeError("Latent statistics are not calibrated")
        labels = self.sample_labels(n, labels)
        z = self.diffusion.sample(n, labels, steps, sampler)
        z = z * self.latent_std + self.latent_mean
        # Deterministic decoded mean; do not silently add Gaussian decoder noise.
        return self.autoencoder.decode(z, labels)[0]

