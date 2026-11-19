"""Small 64px projection-SN cGAN; B/V1/V2 use these identical models.

New implementation of standard residual/cBN/projection components, not an
official SNGAN reproduction. No downloads, training or smoke calls on import.
Labels are ordered group = 2 * Male + Blond_Hair, with annotation bits 0/1.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


DEFAULT_OPTIMIZER_CONFIG = {"lr": 2e-4, "betas": (0.0, 0.9), "eps": 1e-8}


def _initialized(layer, use_sn=False):
    nn.init.orthogonal_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)
    return spectral_norm(layer, n_power_iterations=1, eps=1e-6) if use_sn else layer


def _conv(cin, cout, kernel=3, use_sn=False):
    return _initialized(nn.Conv2d(cin, cout, kernel, padding=kernel // 2), use_sn)


def _check_labels(labels, batch_size):
    if labels.ndim != 1 or labels.shape[0] != batch_size or labels.dtype != torch.long:
        raise ValueError("labels must be a length-batch torch.long tensor")


class ConditionalBatchNorm2d(nn.Module):
    """Shared BN statistics, conditionwise gain=1+offset and bias tables.

    resolution records the actual input feature resolution for loss selection.
    Offsets/biases start at zero; all arms share this initialization. Running
    statistics ARE tracked, so eval-mode generation uses frozen running stats.
    """

    def __init__(self, num_features, num_classes=4, resolution=4):
        super().__init__()
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.resolution = int(resolution)
        self.bn = nn.BatchNorm2d(num_features, affine=False, eps=1e-4, momentum=0.1)
        self.affine = nn.Embedding(num_classes, 2 * num_features)
        nn.init.zeros_(self.affine.weight)

    def conditioning_tables(self):
        """Return gain OFFSET and bias, each [conditions, channels]."""
        return self.affine.weight.split(self.num_features, dim=1)

    def forward(self, x, labels):
        _check_labels(labels, x.shape[0])
        offset, bias = self.affine(labels).split(self.num_features, dim=1)
        return self.bn(x) * (1 + offset[:, :, None, None]) + bias[:, :, None, None]


class GeneratorBlock(nn.Module):
    def __init__(self, cin, cout, num_classes, input_resolution):
        super().__init__()
        self.bn1 = ConditionalBatchNorm2d(cin, num_classes, input_resolution)
        self.bn2 = ConditionalBatchNorm2d(cout, num_classes, 2 * input_resolution)
        self.conv1 = _conv(cin, cout)
        self.conv2 = _conv(cout, cout)
        self.skip = _conv(cin, cout, kernel=1)

    def forward(self, x, labels):
        h = F.relu(self.bn1(x, labels), inplace=False)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = self.conv2(F.relu(self.bn2(h, labels), inplace=False))
        shortcut = self.skip(F.interpolate(x, scale_factor=2, mode="nearest"))
        return h + shortcut


class Generator(nn.Module):
    def __init__(self, z_dim=128, num_classes=4, base_channels=64):
        super().__init__()
        if z_dim < 1 or num_classes < 1 or base_channels < 8 or base_channels % 2:
            raise ValueError("positive dimensions and an even base_channels >= 8 required")
        self.z_dim = int(z_dim)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        c = self.base_channels
        self.linear = _initialized(nn.Linear(self.z_dim, 4 * 4 * 4 * c))
        widths = [4 * c, 4 * c, 2 * c, c, c // 2]
        self.blocks = nn.ModuleList([
            GeneratorBlock(widths[i], widths[i + 1], self.num_classes, 4 * 2**i)
            for i in range(4)
        ])
        self.out_bn = nn.BatchNorm2d(widths[-1], eps=1e-4, momentum=0.1)
        self.to_rgb = _conv(widths[-1], 3)

    def forward(self, z, labels):
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError("z must have shape [batch, z_dim]")
        _check_labels(labels, z.shape[0])
        h = self.linear(z).reshape(z.shape[0], 4 * self.base_channels, 4, 4)
        for block in self.blocks:
            h = block(h, labels)
        return torch.tanh(self.to_rgb(F.relu(self.out_bn(h), inplace=False)))

    def config(self):
        return {"z_dim": self.z_dim, "num_classes": self.num_classes,
                "base_channels": self.base_channels, "image_size": 64,
                "normalization": "tracked_shared_BN_condition_affine",
                "initialization": "orthogonal_convs_linear_zero_cBN_offsets_biases"}


class DiscriminatorBlock(nn.Module):
    def __init__(self, cin, cout, downsample=True, first=False):
        super().__init__()
        self.first = bool(first)
        self.downsample = bool(downsample)
        self.conv1 = _conv(cin, cout, use_sn=True)
        self.conv2 = _conv(cout, cout, use_sn=True)
        self.skip = _conv(cin, cout, kernel=1, use_sn=True) if cin != cout else nn.Identity()

    def forward(self, x):
        h = x if self.first else F.relu(x, inplace=False)
        h = self.conv1(h)
        h = self.conv2(F.relu(h, inplace=False))
        shortcut = self.skip(x)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            shortcut = F.avg_pool2d(shortcut, 2)
        return h + shortcut


class ProjectionDiscriminator(nn.Module):
    def __init__(self, num_classes=4, base_channels=64):
        super().__init__()
        if num_classes < 1 or base_channels < 8 or base_channels % 2:
            raise ValueError("positive num_classes and an even base_channels >= 8 required")
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        c = self.base_channels
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(3, c, first=True),       # 64 -> 32
            DiscriminatorBlock(c, 2 * c),              # 32 -> 16
            DiscriminatorBlock(2 * c, 4 * c),          # 16 -> 8
            DiscriminatorBlock(4 * c, 4 * c),          # 8 -> 4
            DiscriminatorBlock(4 * c, 4 * c, downsample=False),
        ])
        self.linear = _initialized(nn.Linear(4 * c, 1), use_sn=True)
        self.embedding = _initialized(nn.Embedding(self.num_classes, 4 * c), use_sn=True)

    def forward(self, images, labels, return_features=False):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 64, 64):
            raise ValueError("images must have shape [batch, 3, 64, 64]")
        _check_labels(labels, images.shape[0])
        h = images
        for block in self.blocks:
            h = block(h)
        features = F.relu(h, inplace=False).sum(dim=(2, 3))
        scores = self.linear(features).squeeze(1)
        scores = scores + (self.embedding(labels) * features).sum(dim=1)
        return (scores, features) if return_features else scores

    def config(self):
        return {"num_classes": self.num_classes, "base_channels": self.base_channels,
                "image_size": 64, "spectral_normalization": "torch_parametrizations_1_iteration",
                "pooling": "sum", "initialization": "orthogonal_before_SN"}


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def smoke_assertions(device="cpu"):
    """Explicit, tiny forward-only check; never called on module import."""
    torch.manual_seed(123)
    generator = Generator(z_dim=8, base_channels=8).to(device).eval()
    discriminator = ProjectionDiscriminator(base_channels=8).to(device).eval()
    labels = torch.arange(4, device=device, dtype=torch.long)
    with torch.no_grad():
        images = generator(torch.randn(4, 8, device=device), labels)
        scores, features = discriminator(images, labels, return_features=True)
    checks = [images.shape == (4, 3, 64, 64), torch.isfinite(images).all(),
              scores.shape == (4,), torch.isfinite(scores).all(),
              features.shape == (4, 32), images.min() >= -1, images.max() <= 1,
              len([m for m in generator.modules() if isinstance(m, ConditionalBatchNorm2d)]) == 8]
    if not all(bool(check) for check in checks):
        raise AssertionError("model shape/finiteness/range smoke check failed")
    return {"status": "passed", "generator_parameters": parameter_count(generator),
            "discriminator_parameters": parameter_count(discriminator), "device": str(device)}
