"""Gaussian VAE, with optional class conditioning. Primary: arXiv:1312.6114."""
import math
import torch
from .networks import ConditionalMLP, VectorModel


class VAE(VectorModel):
    def __init__(self, dim, width=64, latent_dim=8, classes=0, **unused):
        super().__init__(dim, classes)
        self.latent_dim = int(latent_dim)
        self.encoder = ConditionalMLP(dim, 2 * latent_dim, width, classes)
        self.decoder = ConditionalMLP(latent_dim, 2 * dim, width, classes)

    def encode(self, x, labels=None):
        self.check_x(x, labels)
        mean, logvar = self.encoder(x, labels).chunk(2, -1)
        return mean, logvar.clamp(-8, 4)

    def decode(self, z, labels=None):
        mean, logvar = self.decoder(z, labels).chunk(2, -1)
        return mean, logvar.clamp(-6, 2)

    def loss(self, x, labels=None):
        mean, logvar = self.encode(x, labels)
        z = mean + (0.5 * logvar).exp() * torch.randn_like(mean)
        xmean, xlogvar = self.decode(z, labels)
        # Sum coordinates for the literal Gaussian log likelihood / ELBO.
        nll = 0.5 * (math.log(2 * math.pi) + xlogvar +
                     (x - xmean).square() * (-xlogvar).exp()).sum(-1).mean()
        kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
        return nll + kl, {"negative_log_likelihood": nll.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def sample(self, n, labels=None, steps=None, sampler=None):
        self.no_sampler_options(steps, sampler)
        labels = self.sample_labels(n, labels)
        z = torch.randn(n, self.latent_dim, device=next(self.parameters()).device)
        mean, logvar = self.decode(z, labels)
        return mean + (0.5 * logvar).exp() * torch.randn_like(mean)

