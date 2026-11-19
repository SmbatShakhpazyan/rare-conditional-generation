"""Lagged signed-interaction tracking surrogate; no calls on import.

This proposed algorithm combines known feature matching and tolerance losses.
It is not a new discrepancy or a calibrated confidence procedure. Teacher
features must be computed by a fixed/eval/frozen network while retaining the
gradient through its INPUT to the generator. No trainable parameters are added.
"""
import math

_CACHED_WITNESS_TYPE = None


def _torch():
    import torch
    return torch


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")
    return float(value)


def signed_contrast(features, groups):
    """Image means by group, then (+,-,-,+)/4; differentiable in features."""
    torch = _torch()
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("features must have shape [N,d]")
    if not features.is_floating_point() or features.is_complex():
        raise ValueError("features must be real floating tensors")
    if groups.shape != (features.shape[0],) or groups.dtype != torch.long:
        raise ValueError("groups must be length-N torch.long")
    if groups.device != features.device or bool(((groups < 0) | (groups > 3)).any()):
        raise ValueError("groups must be on feature device and in 0..3")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("nonfinite teacher features")
    x = features.float()
    result = x.new_zeros(x.shape[1])
    for group, coefficient in enumerate((0.25, -0.25, -0.25, 0.25)):
        selected = groups == group
        if not bool(selected.any()):
            raise ValueError("all four groups required in every witness batch")
        result = result + coefficient * x[selected].mean(dim=0)
    return result


def _witness_type():
    global _CACHED_WITNESS_TYPE
    if _CACHED_WITNESS_TYPE is not None:
        return _CACHED_WITNESS_TYPE
    torch = _torch()
    nn = torch.nn
    class _LaggedInteractionWitness(nn.Module):
        """Stateful detached witness; trainer controls explicit post-step updates.

        loss(features,groups,strength) never mutates state or consumes RNG. Return
        loss, diagnostic tensors, and a detached current contrast. Call update on
        that contrast AFTER a successful G step, so the loss always uses previous
        batches. First loss is exactly zero until the tracker is initialized.
        reset_statistics must be called whenever the teacher coordinates change.
        Store config() and state_dict in checkpoints; there are buffers but no
        learnable parameters. The surrogate may be negative: it is not L_dead.
        """

        def __init__(self, statistics, kappa=1.0, beta=0.9, warmup_updates=500):
            super().__init__()
            self.kappa = _finite_nonnegative(kappa, "kappa")
            if isinstance(beta, bool) or not math.isfinite(float(beta)) or not 0 <= beta < 1:
                raise ValueError("beta must be finite in [0,1)")
            if type(warmup_updates) is not int or warmup_updates < 0:
                raise ValueError("warmup_updates must be a nonnegative int")
            self.beta = float(beta)
            self.warmup_updates = warmup_updates
            target, scale, uncertainty = self._vectors(statistics)
            self.register_buffer("target", target)
            self.register_buffer("scale", scale)
            self.register_buffer("uncertainty", uncertainty)
            self.register_buffer("fake_ema", torch.zeros_like(target))
            self.register_buffer("updates", torch.zeros((), dtype=torch.long))

        @staticmethod
        def _vectors(statistics):
            arrays = [torch.as_tensor(statistics[key], dtype=torch.float32).detach().clone()
                      for key in ("target", "scale", "uncertainty")]
            if arrays[0].ndim != 1 or arrays[0].numel() == 0 or any(a.shape != arrays[0].shape for a in arrays):
                raise ValueError("statistics vectors must have equal nonempty one-dimensional shape")
            if not all(bool(torch.isfinite(a).all()) for a in arrays):
                raise ValueError("nonfinite statistics")
            if not bool((arrays[1] > 0).all()) or not bool((arrays[2] >= 0).all()):
                raise ValueError("scale positive and uncertainty nonnegative required")
            return arrays

        def config(self):
            return {"kappa": self.kappa, "beta": self.beta,
                    "warmup_updates": self.warmup_updates,
                    "feature_dim": self.target.numel(),
                    "contrast": [0.25, -0.25, -0.25, 0.25],
                    "objective": "lagged_detached_deadzone_witness_linear_surrogate",
                    "tracker_initialization": "first_successful_post_step_contrast",
                    "confidence_bound": False}

        @torch.no_grad()
        def reset_statistics(self, statistics):
            arrays = self._vectors(statistics)
            if arrays[0].shape != self.target.shape:
                raise ValueError("teacher refresh must preserve feature dimension")
            for destination, value in zip((self.target, self.scale, self.uncertainty), arrays):
                destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
            self.fake_ema.zero_()
            self.updates.zero_()

        @torch.no_grad()
        def update(self, detached_contrast):
            """Explicit call AFTER successful G optimizer step; no update on AMP skip."""
            if detached_contrast.requires_grad:
                raise ValueError("tracker update requires an explicitly detached contrast")
            if detached_contrast.shape != self.target.shape or detached_contrast.device != self.target.device:
                raise ValueError("tracker contrast shape/device mismatch")
            value = detached_contrast.to(dtype=self.fake_ema.dtype)
            if not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite tracker contrast")
            if int(self.updates.item()) == 0:
                self.fake_ema.copy_(value)
            else:
                self.fake_ema.mul_(self.beta).add_(value, alpha=1 - self.beta)
            self.updates.add_(1)

        def loss(self, features, groups, strength=1.0, kappa=None):
            strength = _finite_nonnegative(strength, "strength")
            actual_kappa = self.kappa if kappa is None else _finite_nonnegative(kappa, "kappa")
            current = signed_contrast(features, groups)
            if current.shape != self.target.shape or current.device != self.target.device:
                raise ValueError("features do not match witness statistics shape/device")
            with torch.no_grad():
                previous = (self.fake_ema - self.target) / self.scale
                tolerance = actual_kappa * self.uncertainty / self.scale
                witness = torch.sign(previous) * torch.relu(previous.abs() - tolerance)
                ready = int(self.updates.item()) > 0 and int(self.updates.item()) >= self.warmup_updates
                if not ready:
                    witness.zero_()
                now_standardized = (current.detach() - self.target) / self.scale
                current_deadzone = torch.relu(now_standardized.abs() - tolerance).square().mean()
            # No current-batch residual appears in the derivative coefficient.
            result = (2.0 * strength / current.numel()) * torch.sum(witness * (current - self.target) / self.scale)
            diagnostics = {"witness_active_fraction": (witness != 0).float().mean(),
                           "witness_rms": witness.square().mean().sqrt(),
                           "current_deadzone_descriptive_only": current_deadzone,
                           "tracker_updates": self.updates.detach().clone()}
            return result, diagnostics, current.detach()
    _CACHED_WITNESS_TYPE = _LaggedInteractionWitness
    return _CACHED_WITNESS_TYPE


def LaggedInteractionWitness(statistics, kappa=1.0, beta=0.9, warmup_updates=500):
    """Lazy Torch factory returning nn.Module; checkpoint state_dict, not object."""
    return _witness_type()(statistics, kappa=kappa, beta=beta, warmup_updates=warmup_updates)


Witness = LaggedInteractionWitness
