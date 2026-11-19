"""Matched baseline, capped critic-gap weighting and cBN regularizers.

No model construction/training on import. gSR is a documented exact-SVD
adaptation of the authors' supported SVD branch, not their full training recipe.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


GSR_AVAILABLE = True
GSR_IMPLEMENTATION = "author_supported_exact_SVD_branch_same_backbone_adaptation"


def _score_vector(scores):
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a nonempty [batch] or [batch, 1] tensor")
    return scores.float()


def per_group_mean(scores, labels, num_classes=4):
    """Every group must occur; use batch reduction for natural-prior context."""
    scores = _score_vector(scores)
    if labels.ndim != 1 or labels.shape[0] != scores.shape[0] or labels.dtype != torch.long:
        raise ValueError("labels must be a matching torch.long vector")
    if labels.device != scores.device:
        raise ValueError("scores and labels must be on the same device")
    counts = torch.bincount(labels, minlength=num_classes)
    if counts.numel() != num_classes or bool((counts == 0).any()):
        raise ValueError("balanced loss requires at least one example from every group")
    totals = scores.new_zeros(num_classes).scatter_add(0, labels, scores)
    return totals / counts.to(dtype=totals.dtype)


def hinge_discriminator_loss(real_scores, fake_scores, real_labels=None,
                             fake_labels=None, num_classes=4, group_balanced=True):
    real = F.relu(1 - _score_vector(real_scores))
    fake = F.relu(1 + _score_vector(fake_scores))
    if not group_balanced:
        return real.mean() + fake.mean()
    if real_labels is None or fake_labels is None:
        raise ValueError("balanced discriminator loss needs real and fake labels")
    return (per_group_mean(real, real_labels, num_classes).mean()
            + per_group_mean(fake, fake_labels, num_classes).mean())


def generator_adversarial_loss(fake_scores, labels=None, weights=None,
                               num_classes=4, group_balanced=True):
    scores = _score_vector(fake_scores)
    if not group_balanced:
        if weights is not None:
            raise ValueError("natural-prior batch reduction does not take group weights")
        return -scores.mean()
    if labels is None:
        raise ValueError("balanced generator loss needs labels")
    means = per_group_mean(-scores, labels, num_classes)
    if weights is None:
        weights = means.new_full((num_classes,), 1.0 / num_classes)
    else:
        weights = weights.detach().to(device=means.device, dtype=means.dtype)
        if weights.shape != (num_classes,) or not bool(torch.isfinite(weights).all()):
            raise ValueError("group weights must be a finite length-num_classes vector")
        if bool((weights < 0).any()) or not bool(torch.isclose(
                weights.sum(), weights.new_tensor(1.0), rtol=1e-6, atol=1e-7)):
            raise ValueError("group weights must be nonnegative and sum to one")
    # Baseline and explicit uniform weights intentionally use the same operation.
    return (weights * means).sum()


class CriticGapWeighter(nn.Module):
    """Checkpointable NONTRAINABLE V1 state.

    Call weights() for the G update BEFORE update(...) from that iteration's
    detached D-step scores. This enforces the predeclared one-step lag. State
    update never consumes random numbers. Save config() alongside state_dict().
    """

    def __init__(self, num_classes=4, strength=1.0 / 3.0, temperature=1.0,
                 ema_decay=0.99, gap_clip=4.0, warmup_steps=1000):
        super().__init__()
        if not all(math.isfinite(float(value)) for value in
                   [strength, temperature, ema_decay, gap_clip]):
            raise ValueError("V1 hyperparameters must be finite")
        if num_classes < 1 or not 0 <= strength <= 1 or temperature <= 0:
            raise ValueError("invalid classes, strength or temperature")
        if not 0 <= ema_decay < 1 or gap_clip <= 0 or warmup_steps < 0:
            raise ValueError("invalid EMA, clip or warmup")
        self.num_classes = int(num_classes)
        self.strength = float(strength)
        self.temperature = float(temperature)
        self.ema_decay = float(ema_decay)
        self.gap_clip = float(gap_clip)
        self.warmup_steps = int(warmup_steps)
        self.register_buffer("gap_ema", torch.zeros(self.num_classes))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def config(self):
        return {"num_classes": self.num_classes, "strength": self.strength,
                "temperature": self.temperature, "ema_decay": self.ema_decay,
                "gap_clip": self.gap_clip, "warmup_steps": self.warmup_steps}

    def weights(self):
        uniform = self.gap_ema.new_full((self.num_classes,), 1.0 / self.num_classes)
        if self.strength == 0:
            return uniform
        adaptive = ((1 - self.strength) * uniform
                    + self.strength * torch.softmax(self.gap_ema / self.temperature, dim=0))
        return torch.where(self.num_updates < self.warmup_steps, uniform, adaptive).detach()

    @torch.no_grad()
    def update(self, real_scores, fake_scores, real_labels, fake_labels):
        gap = (per_group_mean(real_scores.detach(), real_labels, self.num_classes)
               - per_group_mean(fake_scores.detach(), fake_labels, self.num_classes))
        if gap.device != self.gap_ema.device:
            raise ValueError("move CriticGapWeighter to the score device before updating")
        if not bool(torch.isfinite(gap).all()):
            raise FloatingPointError("nonfinite critic gap")
        self.gap_ema.mul_(self.ema_decay).add_(gap.clamp(0, self.gap_clip),
                                            alpha=1 - self.ema_decay)
        self.num_updates.add_(1)
        return {"raw_gap": gap.clone(), "gap_ema": self.gap_ema.clone(),
                "next_weights": self.weights().clone()}


def _conditional_layers(generator):
    # Duck typing keeps these losses usable with an audited matching backbone.
    return [(name, layer) for name, layer in generator.named_modules()
            if callable(getattr(layer, "conditioning_tables", None))]


def coarse_interaction_penalty(generator, max_resolution=16):
    terms = []
    for name, layer in _conditional_layers(generator):
        if layer.resolution > max_resolution:
            continue
        offset, bias = layer.conditioning_tables()
        if offset.shape[0] != 4:
            raise ValueError("factorial penalty requires group=2*bitA+bitB and four conditions")
        interaction_gain = (offset[0] - offset[1] - offset[2] + offset[3]) / 4
        interaction_bias = (bias[0] - bias[1] - bias[2] + bias[3]) / 4
        terms.append(interaction_gain.float().square().mean()
                     + interaction_bias.float().square().mean())
    if not terms:
        raise ValueError("no conditional layers at the requested coarse resolutions")
    return torch.stack(terms).mean()


def coarse_interaction_layer_names(generator, max_resolution=16):
    return [name for name, layer in _conditional_layers(generator)
            if layer.resolution <= max_resolution]


def _balanced_factors(width):
    rows = math.isqrt(width)
    while width % rows:
        rows -= 1
    return rows, width // rows


def group_spectral_regularization(generator, class_counts, beta=0.999,
                                  regularizer_weight=0.5):
    """gSR exact-SVD adaptation, including author's effective-weight normalization.

    Author source: val-iisc/gSRGAN src/utils/model_ops.py (SVD branch,
    gain offset/bias grouping) and src/worker.py (class/layer normalization).
    Return includes regularizer_weight: add it ONCE, not twice, to G loss.
    No LeCam/DiffAug, five-D-step schedule or other author recipe is implied.
    """
    if not math.isfinite(float(regularizer_weight)) or not 0 <= beta < 1 or regularizer_weight < 0:
        raise ValueError("beta must be in [0,1), regularizer_weight nonnegative")
    layers = _conditional_layers(generator)
    if not layers:
        raise ValueError("gSR requires conditional affine layers")
    first_table = layers[0][1].conditioning_tables()[0]
    if regularizer_weight == 0:
        return first_table.new_zeros((), dtype=torch.float32)
    counts = torch.as_tensor(class_counts, device=first_table.device, dtype=torch.float64)
    if counts.shape != (first_table.shape[0],) or not bool(torch.isfinite(counts).all()):
        raise ValueError("one finite positive training count per condition required")
    if bool((counts <= 0).any()):
        raise ValueError("gSR training class counts must be strictly positive")
    # Inverse effective counts, rescaled to mean class weight one as author code.
    inverse_effective = (torch.ones_like(counts) if beta == 0 else
                         (1 - beta) / (-torch.expm1(counts * math.log(beta))))
    class_weights = inverse_effective / inverse_effective.mean()
    terms = []
    # Disable autocast: half-precision SVD is unsupported and unstable here.
    with torch.autocast(device_type=first_table.device.type, enabled=False):
        for _, layer in layers:
            offset, bias = layer.conditioning_tables()
            if offset.shape[0] != counts.numel():
                raise ValueError("conditional layer class counts disagree")
            rows, cols = _balanced_factors(offset.shape[1])
            # Double precision matches the authors' exact-SVD branch. Their
            # gain parameter is the offset in gain=1+offset, NOT the added one.
            sigma_gain = torch.linalg.svdvals(offset.double().reshape(-1, rows, cols))[:, 0]
            sigma_bias = torch.linalg.svdvals(bias.double().reshape(-1, rows, cols))[:, 0]
            terms.append(((sigma_gain.square() + sigma_bias.square()) * class_weights).mean())
        return (regularizer_weight * torch.stack(terms).mean()).float()


def generator_variant_loss(fake_scores, labels, generator, variant="B", weighter=None,
                           interaction_strength=1.0, max_resolution=16,
                           class_counts=None, gsr_weight=0.5, gsr_beta=0.999):
    """Common entry point returning (total_loss, detached scalar/tensor terms)."""
    if variant not in {"B", "V1", "V2", "gSR", "natural"}:
        raise ValueError("unknown variant")
    if not math.isfinite(float(interaction_strength)) or interaction_strength < 0:
        raise ValueError("interaction_strength must be nonnegative")
    weights = None
    if variant == "V1":
        if weighter is None:
            raise ValueError("V1 requires a CriticGapWeighter")
        weights = weighter.weights()
    adversarial = generator_adversarial_loss(fake_scores, labels, weights=weights,
                                             group_balanced=variant != "natural")
    total = adversarial
    regularizer = adversarial.new_zeros(())
    if variant == "V2" and interaction_strength != 0:
        regularizer = interaction_strength * coarse_interaction_penalty(generator, max_resolution)
        total = total + regularizer
    elif variant == "gSR":
        if class_counts is None:
            raise ValueError("gSR requires the actual training class_counts")
        regularizer = group_spectral_regularization(generator, class_counts, gsr_beta, gsr_weight)
        total = total + regularizer
    return total, {"adversarial": adversarial.detach(), "regularizer": regularizer.detach(),
                   "weights": None if weights is None else weights.detach()}


def smoke_assertions():
    """Explicit tiny tensor-only mechanism checks; no model or optimizer training."""
    labels = torch.arange(4, dtype=torch.long)
    real = torch.tensor([1.0, 2.0, 3.0, 4.0])
    fake = -real
    baseline = generator_adversarial_loss(fake, labels)
    def require(condition, label):
        if not bool(condition):
            raise AssertionError(label)

    require(torch.equal(baseline, generator_adversarial_loss(fake, labels, torch.full((4,), 0.25))),
            "explicit uniform weights differ from baseline")
    require(hinge_discriminator_loss(real, fake, labels, labels).item() == 0, "hinge margins")
    disabled = CriticGapWeighter(strength=0, warmup_steps=0)
    disabled.update(real, fake, labels, labels)
    require(torch.equal(disabled.weights(), torch.full((4,), 0.25)), "disabled V1 weights")
    adaptive = CriticGapWeighter(ema_decay=0, warmup_steps=0)
    adaptive.update(torch.tensor([0.0, 1.0, 2.0, 4.0]), torch.zeros(4), labels, labels)
    weights = adaptive.weights()
    require((weights.min() >= 1.0 / 6.0) & (weights.max() <= 0.5), "V1 cap/floor")
    require(torch.isclose(weights.sum(), torch.tensor(1.0)), "V1 weight sum")
    restored = CriticGapWeighter(ema_decay=0, warmup_steps=0)
    restored.load_state_dict(adaptive.state_dict())
    require(torch.equal(restored.weights(), weights), "V1 state restoration")
    # A class-specific additive D offset must leave the measured gap unchanged.
    shifted = CriticGapWeighter(ema_decay=0, warmup_steps=0)
    shifts = torch.tensor([10.0, -2.0, 1.0, 8.0])
    shifted.update(torch.tensor([0.0, 1.0, 2.0, 4.0]) + shifts, shifts, labels, labels)
    require(torch.equal(shifted.weights(), weights), "condition-offset invariance")

    class ToyCondition(nn.Module):
        def __init__(self):
            super().__init__()
            self.resolution = 4
            self.offset = nn.Parameter(torch.tensor([[0., 0.], [1., 1.], [2., 2.], [3., 3.]]))
            self.bias = nn.Parameter(torch.zeros(4, 2))

        def conditioning_tables(self):
            return self.offset, self.bias

    toy = nn.Sequential(ToyCondition())
    require(coarse_interaction_penalty(toy).item() == 0, "additive tables must have zero interaction")
    with torch.no_grad():
        toy[0].offset[3].add_(4)
    require(coarse_interaction_penalty(toy).item() == 1, "interaction square/normalization")
    disabled_v2, _ = generator_variant_loss(fake, labels, toy, variant="V2", interaction_strength=0)
    require(torch.equal(disabled_v2, baseline), "disabled V2 differs from baseline")
    penalty = group_spectral_regularization(toy, [10, 20, 30, 40])
    penalty.backward()
    require(torch.isfinite(penalty), "gSR finite forward")
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in toy.parameters()),
            "gSR finite gradients, including zero-bias table")
    return {"status": "passed", "baseline_generator_loss": baseline.item(),
            "gSR": GSR_IMPLEMENTATION}
