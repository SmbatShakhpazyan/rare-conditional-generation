"""Explicit algorithm registry and serializable tiny reference presets."""
from dataclasses import asdict, dataclass
import math
from .vae import VAE
from .cgan import CGAN
from .autoregressive import AutoregressiveGaussian
from .flows import RealNVP
from .diffusion import DDPM
from .score import NoiseScore, VESDE
from .flow_matching import FlowMatching
from .consistency import Consistency
from .latent_diffusion import LatentDiffusion

MODELS = {"vae": VAE, "cgan": CGAN, "autoregressive": AutoregressiveGaussian,
          "realnvp": RealNVP, "ddpm": DDPM, "score": NoiseScore, "sde": VESDE,
          "flow_matching": FlowMatching, "consistency": Consistency,
          "latent_diffusion": LatentDiffusion}


@dataclass(frozen=True)
class Config:
    dim: int
    classes: int = 0
    width: int = 64
    latent_dim: int = 8
    coupling_layers: int = 4
    diffusion_steps: int = 32
    beta_start: float = 1e-4
    beta_end: float = .2
    sigma_min: float = .01
    sigma_max: float = 2.
    noise_levels: int = 8
    score_step_size: float = 2e-5
    consistency_ema: float = .95

    def validate(self):
        for name in ("dim", "width", "latent_dim", "coupling_layers", "diffusion_steps", "noise_levels"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.classes) is not int or self.classes < 0:
            raise ValueError("classes must be a nonnegative integer")
        for name in ("beta_start", "beta_end", "sigma_min", "sigma_max",
                     "score_step_size", "consistency_ema"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"Nonfinite {name}")
        if self.coupling_layers < 2 or self.diffusion_steps < 2 or self.noise_levels < 2:
            raise ValueError("At least two couplings / noise levels / diffusion steps required")
        if not 0 < self.beta_start <= self.beta_end < 1:
            raise ValueError("Invalid beta range")
        if not 0 < self.sigma_min < self.sigma_max or self.score_step_size <= 0:
            raise ValueError("Invalid noise range or Langevin step")
        if not 0 <= self.consistency_ema < 1:
            raise ValueError("Invalid consistency EMA")
        return self


def preset(name, dim, classes=0):
    if name == "tiny":
        return Config(dim=dim, classes=classes).validate()
    if name == "small":
        return Config(dim=dim, classes=classes, width=128, latent_dim=16,
                      coupling_layers=6, diffusion_steps=128, beta_end=.05,
                      noise_levels=16).validate()
    raise ValueError("Choose preset tiny or small")


def create(name, config):
    if name not in MODELS:
        raise ValueError("Unknown reference algorithm")
    if isinstance(config, dict):
        config = Config(**config)
    config.validate()
    return MODELS[name](**asdict(config))

