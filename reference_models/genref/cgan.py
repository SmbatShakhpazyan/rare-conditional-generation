"""Projection hinge cGAN; original small MLP architecture.

Conditional GAN: arXiv:1411.1784; projection: arXiv:1802.05637;
spectral normalization: arXiv:1802.05957. Not an official SNGAN replica.
"""
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from .networks import ConditionalMLP, VectorModel, labels_for


class ProjectionDiscriminator(nn.Module):
    def __init__(self, dim, width, classes):
        super().__init__()
        sn = spectral_norm
        self.classes = classes
        self.features = nn.Sequential(sn(nn.Linear(dim, width)), nn.LeakyReLU(.2),
                                      sn(nn.Linear(width, width)), nn.LeakyReLU(.2))
        self.head = sn(nn.Linear(width, 1))
        self.embedding = sn(nn.Embedding(classes, width)) if classes else None

    def forward(self, x, labels=None):
        labels = labels_for(labels, len(x), self.classes, x.device)
        h = self.features(x)
        out = self.head(h).squeeze(-1)
        if self.embedding is not None:
            out = out + (h * self.embedding(labels)).sum(-1)
        return out


class CGAN(VectorModel):
    def __init__(self, dim, width=64, latent_dim=8, classes=0, **unused):
        super().__init__(dim, classes)
        self.latent_dim = int(latent_dim)
        self.generator = ConditionalMLP(latent_dim, dim, width, classes, depth=3)
        self.discriminator = ProjectionDiscriminator(dim, width, classes)

    def generate(self, z, labels=None):
        return self.generator(z, labels).tanh()

    def discriminator_loss(self, x, labels=None):
        self.check_x(x, labels)
        z = torch.randn(len(x), self.latent_dim, device=x.device)
        fake = self.generate(z, labels).detach()
        real_score = self.discriminator(x, labels)
        fake_score = self.discriminator(fake, labels)
        loss = (1 - real_score).relu().mean() + (1 + fake_score).relu().mean()
        return loss, {"real_score": real_score.mean().detach(),
                      "fake_score": fake_score.mean().detach()}

    def generator_loss(self, n, labels=None):
        z = torch.randn(n, self.latent_dim, device=next(self.parameters()).device)
        loss = -self.discriminator(self.generate(z, labels), labels).mean()
        return loss, {"generator_adversarial": loss.detach()}

    @torch.no_grad()
    def sample(self, n, labels=None, steps=None, sampler=None):
        self.no_sampler_options(steps, sampler)
        labels = self.sample_labels(n, labels)
        z = torch.randn(n, self.latent_dim, device=next(self.parameters()).device)
        return self.generate(z, labels)

