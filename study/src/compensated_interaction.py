"""Local positive-channel-gauge-compensated cBN interaction penalty.

This module has no torch import, RNG use, device initialization, model
construction or training on import. The penalty is invariant to the specified
cBN-affine/next-convolution positive channel rescaling, not to every
function-preserving reparameterization. It is not a GAN-quality guarantee.
"""
from __future__ import annotations


COMPENSATED_INTERACTION_IMPLEMENTATION = (
    "coarse_hadamard_interaction_times_next_conv_input_channel_frobenius_squared_v1"
)


def compensated_interaction_pairs(generator, max_resolution=16):
    """Return explicit (cBN name, cBN, conv name, conv) pairs in block order.

    This deliberately recognizes the audited GeneratorBlock topology only:
    blocks.N.bn1 -> blocks.N.conv1 and blocks.N.bn2 -> blocks.N.conv2.
    No inferred nearest-layer matching or forward hooks are used.
    """
    if isinstance(max_resolution, bool) or not isinstance(max_resolution, int) or max_resolution < 1:
        raise ValueError("max_resolution must be a positive integer")
    blocks = getattr(generator, "blocks", None)
    if blocks is None:
        raise ValueError("expected the audited generator.blocks topology")
    pairs = []
    for index, block in enumerate(blocks):
        for suffix in ("1", "2"):
            cbn_name = f"blocks.{index}.bn{suffix}"
            conv_name = f"blocks.{index}.conv{suffix}"
            cbn = getattr(block, "bn" + suffix, None)
            if not callable(getattr(cbn, "conditioning_tables", None)):
                raise ValueError(f"{cbn_name} must expose conditioning_tables()")
            resolution = getattr(cbn, "resolution", None)
            if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 1:
                raise ValueError(f"{cbn_name} must record a positive integer input resolution")
            if resolution > max_resolution:
                continue
            conv = getattr(block, "conv" + suffix, None)
            if conv is None or getattr(conv, "groups", None) != 1:
                raise ValueError(f"{conv_name} must be the following dense-channel Conv2d")
            pairs.append((cbn_name, cbn, conv_name, conv))
    if not pairs:
        raise ValueError("no cBN/convolution pairs at the requested coarse resolutions")
    return pairs


def compensated_interaction_layer_names(generator, max_resolution=16):
    """Names of selected cBN modules; five for the common 64px backbone."""
    return [name for name, _, _, _ in compensated_interaction_pairs(generator, max_resolution)]


def compensated_interaction_penalty(generator, max_resolution=16):
    """FP32 scalar: mean_l mean_c ||W_l[:,c,:,:]||_F^2 (I_gamma^2+I_beta^2).

    I_t = (t00-t01-t10+t11)/4, with row = 2*bitA+bitB. Gamma is the
    effective gain 1+offset; its contrast equals the offset contrast.
    Gradients flow through BOTH the tables and convolution weights. There
    is no detach, stabilizing epsilon, extra normalization, RNG or mutation.
    The caller adds its declared coefficient (default experimental lambda=1)
    exactly once. This function does not include that coefficient.
    """
    import torch

    terms = []
    for cbn_name, cbn, conv_name, conv in compensated_interaction_pairs(generator, max_resolution):
        offset, bias = cbn.conditioning_tables()
        weight = conv.weight
        if offset.ndim != 2 or offset.shape[0] != 4 or bias.shape != offset.shape:
            raise ValueError(f"{cbn_name} must have four matching gain-offset/bias rows")
        if weight.ndim != 4 or weight.shape[1] != offset.shape[1]:
            raise ValueError(f"{conv_name} input channels must match {cbn_name}")
        if offset.device != bias.device or offset.device != weight.device:
            raise ValueError(f"{cbn_name} and {conv_name} must use the same device")
        with torch.autocast(device_type=weight.device.type, enabled=False):
            gain_table = offset.float()
            bias_table = bias.float()
            interaction_gain = (gain_table[0] - gain_table[1] - gain_table[2] + gain_table[3]) / 4
            interaction_bias = (bias_table[0] - bias_table[1] - bias_table[2] + bias_table[3]) / 4
            next_weight_norm_squared = weight.float().square().sum(dim=(0, 2, 3))
            terms.append((next_weight_norm_squared * (
                interaction_gain.square() + interaction_bias.square()
            )).mean())
    return torch.stack(terms).mean()


def smoke_assertions(device="cpu", seed=1729, atol=1e-4, base_channels=64):
    """Explicit untrained FULL Generator checks, intended for a torch runtime.

    Checks all four labels, nonzero affines/conv biases/BN running statistics,
    eval and train modes, scalar c=.1 and per-channel c in [.3,2], old V2
    shrinkage, compensated invariance, and finite nonzero gradients. No dataset,
    optimizer, checkpoint, pretrained weights, or training step is used. The
    caller must invoke this function; importing the module does not run it.
    """
    import copy
    from contextlib import contextmanager
    import torch
    from torch import nn

    if __package__:
        from .models import Generator
        from .model_losses import coarse_interaction_penalty
    else:
        from models import Generator
        from model_losses import coarse_interaction_penalty

    target_device = torch.device(device)
    cuda_devices = []
    if target_device.type == "cuda":
        cuda_devices = [torch.cuda.current_device() if target_device.index is None else target_device.index]

    def require(condition, message):
        if not bool(condition):
            raise AssertionError(message)

    def apply_rescaling(model, mode):
        with torch.no_grad():
            for _, cbn, _, conv in compensated_interaction_pairs(model):
                offset, bias = cbn.conditioning_tables()
                if mode == "uniform_0.1":
                    scales = offset.new_full((offset.shape[1],), .1)
                else:
                    scales = torch.linspace(.3, 2., offset.shape[1], device=offset.device,
                                            dtype=offset.dtype)
                offset.copy_(scales[None, :] * (1 + offset) - 1)
                bias.mul_(scales[None, :])
                conv.weight.div_(scales[None, :, None, None])

    def bn_buffers(model):
        return {name: value.detach().clone() for name, value in model.named_buffers()
                if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}

    @contextmanager
    def strict_float32():
        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            torch.backends.cudnn.allow_tf32 = previous_cudnn

    with torch.random.fork_rng(devices=cuda_devices), strict_float32(), \
            torch.autocast(device_type=target_device.type, enabled=False):
        # Seed only the forked CPU generator used by model initialization.
        # Tensor fixtures use a private generator, never manual_seed-all.
        torch.random.default_generator.manual_seed(seed)
        fixture_rng = torch.Generator(device=target_device).manual_seed(seed + 1)
        generator = Generator(base_channels=base_channels).to(target_device).float().eval()
        require(compensated_interaction_penalty(generator).item() == 0,
                "zero-initialized cBN tables must have zero compensated penalty")
        expected_names = ["blocks.0.bn1", "blocks.0.bn2", "blocks.1.bn1",
                          "blocks.1.bn2", "blocks.2.bn1"]
        require(compensated_interaction_layer_names(generator) == expected_names,
                "unexpected selected cBN modules")
        with torch.no_grad():
            for module in generator.modules():
                if callable(getattr(module, "conditioning_tables", None)):
                    module.affine.weight.normal_(mean=.03, std=.15, generator=fixture_rng)
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean.normal_(mean=.03, std=.04, generator=fixture_rng)
                    module.running_var.uniform_(.7, 1.4, generator=fixture_rng)
            for _, _, _, conv in compensated_interaction_pairs(generator):
                if conv.bias is not None:
                    conv.bias.normal_(mean=.02, std=.03, generator=fixture_rng)
        labels = torch.arange(4, device=target_device, dtype=torch.long)
        noise = torch.randn(4, generator.z_dim, device=target_device, generator=fixture_rng)
        old_before = coarse_interaction_penalty(generator).detach()
        new_before = compensated_interaction_penalty(generator).detach()
        require(old_before > 0, "fixture needs nonzero original interaction penalty")
        require(new_before > 0, "fixture needs nonzero compensated penalty")
        scenarios = []
        for mode in ("uniform_0.1", "per_channel_0.3_to_2.0"):
            rescaled = copy.deepcopy(generator)
            unchanged_buffers = bn_buffers(rescaled)
            apply_rescaling(rescaled, mode)
            require(all(torch.equal(value, bn_buffers(rescaled)[name])
                        for name, value in unchanged_buffers.items()),
                    "rescaling must not change BN buffers")
            old_after = coarse_interaction_penalty(rescaled).detach()
            new_after = compensated_interaction_penalty(rescaled).detach()
            require(torch.allclose(new_before, new_after, rtol=1e-5, atol=1e-7),
                    "compensated penalty changed under local channel rescaling")
            if mode == "uniform_0.1":
                require(torch.allclose(old_after, old_before * .01, rtol=1e-5, atol=1e-8),
                        "old interaction penalty must shrink by c squared")
            mode_result = {"rescaling": mode, "old_before": float(old_before),
                           "old_after": float(old_after), "new_before": float(new_before),
                           "new_after": float(new_after), "forward_checks": []}
            for training in (False, True):
                reference = copy.deepcopy(generator).train(training)
                changed = copy.deepcopy(rescaled).train(training)
                with torch.no_grad():
                    reference_images = reference(noise, labels)
                    changed_images = changed(noise, labels)
                error = (reference_images - changed_images).abs().max()
                require(torch.isfinite(reference_images).all() & torch.isfinite(changed_images).all(),
                        "nonfinite full-model fixture output")
                require(error <= atol, "full Generator output exceeded absolute tolerance")
                reference_buffers, changed_buffers = bn_buffers(reference), bn_buffers(changed)
                require(all(torch.allclose(value, changed_buffers[name], rtol=1e-5, atol=1e-6)
                            for name, value in reference_buffers.items()),
                        "BN running-state updates differ beyond float32 tolerance")
                mode_result["forward_checks"].append(
                    {"training_mode": training, "max_absolute_image_difference": float(error)})
            scenarios.append(mode_result)

        generator.zero_grad(set_to_none=True)
        penalty = compensated_interaction_penalty(generator)
        require(penalty.dtype == torch.float32, "penalty must return FP32")
        penalty.backward()
        for name, cbn, conv_name, conv in compensated_interaction_pairs(generator):
            for parameter_name, parameter in ((name + ".affine.weight", cbn.affine.weight),
                                               (conv_name + ".weight", conv.weight)):
                require(parameter.grad is not None and torch.isfinite(parameter.grad).all(),
                        f"missing/nonfinite gradient for {parameter_name}")
                require(parameter.grad.abs().sum() > 0, f"zero fixture gradient for {parameter_name}")
            require(conv.bias is None or conv.bias.grad is None,
                    "paired convolution additive bias must not enter the penalty")
        return {"status": "passed", "scope": "untrained full Generator, no optimizer or data",
                "implementation": COMPENSATED_INTERACTION_IMPLEMENTATION,
                "device": str(target_device), "seed": seed, "atol": atol,
                "base_channels": base_channels, "tf32_during_fixture": False,
                "selected_layers": expected_names, "scenarios": scenarios,
                "finite_nonzero_table_and_convolution_gradients": True}
