"""Independent endpoint straight-path conditional flow matching.

Primary: arXiv:2210.02747. The endpoint interpolation is not an assertion of
globally optimal transport coupling. Sampling uses explicit Euler or Heun.
"""
import torch
from .networks import TimeMLP, VectorModel


class FlowMatching(VectorModel):
    def __init__(self, dim, width=64, classes=0, **unused):
        super().__init__(dim, classes)
        self.velocity = TimeMLP(dim, width, classes)

    def loss(self, x, labels=None):
        self.check_x(x, labels)
        base = torch.randn_like(x)
        time = torch.rand(len(x), device=x.device)
        path = (1 - time[:, None]) * base + time[:, None] * x
        value = (self.velocity(path, time, labels) - (x - base)).square().mean()
        return value, {"conditional_velocity_mse": value.detach()}

    @torch.no_grad()
    def sample(self, n, labels=None, steps=None, sampler=None):
        count = 32 if steps is None else int(steps)
        sampler = "heun" if sampler is None else sampler
        if count < 1 or sampler not in ("euler", "heun"):
            raise ValueError("Choose positive steps and sampler euler or heun")
        labels = self.sample_labels(n, labels)
        x = torch.randn(n, self.dim, device=next(self.parameters()).device)
        h = 1. / count
        for i in range(count):
            time = x.new_full((n,), i * h)
            first = self.velocity(x, time, labels)
            if sampler == "euler":
                x = x + h * first
            else:
                second = self.velocity(x + h * first, time + h, labels)
                x = x + .5 * h * (first + second)
        return x

