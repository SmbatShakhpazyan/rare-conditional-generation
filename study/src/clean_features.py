"""Author-code Clean-FID mode='clean' Inception features for uint8 RGB batches.

No Torch/cleanfid import, network operation or model load occurs on import.
The parent may use this primary only after a pre-main cloud smoke succeeds.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CleanFIDFeatures:
    """Same resize/model path as author build_resizer('clean') + clean model.

    Input: uint8 BHWC RGB (normally the shared64px cache/generated images).
    Output: N-by2048 NumPy float64, preserving every input row.
    Download is explicit; the actual complete TorchScript-file SHA is recorded.
    """
    def __init__(self, device="cuda", weights_dir="/tmp", allow_download=False,
                 expected_weights_sha256=None, expected_package_version="0.1.35"):
        import torch
        import torchvision
        import PIL
        from cleanfid import features, resize, inception_torchscript, downloads_helper, utils

        package_version = version("clean-fid")
        if expected_package_version is not None and package_version != expected_package_version:
            raise ValueError(f"Expected clean-fid{expected_package_version}, found{package_version}")
        directory = Path(weights_dir)
        path = directory / "inception-2015-12-05.pt"
        if not path.is_file():
            if not allow_download:
                raise FileNotFoundError("Clean-FID weights absent; set allow_download=True explicitly")
            directory.mkdir(parents=True, exist_ok=True)
            downloads_helper.check_download_inception(fpath=str(directory))
        actual_sha = _sha256(path)
        if expected_weights_sha256 is not None and actual_sha != expected_weights_sha256:
            raise ValueError("Clean-FID TorchScript weights SHA256 differs")
        # This is precisely the author's features.py clean branch, using the
        # inspectable module instead of its equivalent closure/DataParallel.
        self.model = inception_torchscript.InceptionV3W(
            str(directory), download=False, resize_inside=False,
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.resize = resize.build_resizer("clean")
        self.to_tensor = torchvision.transforms.ToTensor()
        self.torch, self.device = torch, device
        sources = {name: _sha256(module.__file__) for name, module in {
            "features.py": features, "resize.py": resize,
            "inception_torchscript.py": inception_torchscript,
            "downloads_helper.py": downloads_helper, "utils.py": utils,
        }.items()}
        self.provenance = {
            "feature_id": "clean_fid_mode_clean_torchscript_inception_2048",
            "standard_cleanfid": True, "mode": "clean", "feature_dim": 2048,
            "implementation_version": package_version,
            "torch_version": str(torch.__version__),
            "torchvision_version": str(torchvision.__version__), "pillow_version": PIL.__version__,
            "numpy_version": np.__version__, "weights_sha256": actual_sha,
            "weights_file_name": path.name, "weights_url": downloads_helper.inception_url,
            "author_source_sha256": sources, "adapter_source_sha256": _sha256(__file__),
            "preprocessing": {
                "input": "uint8_BHWC_RGB_0_255",
                "resize": "author_build_resizer_clean_PIL_float32_F_per_channel_bicubic299x299_clip0_255",
                "quantize_after_resize": False,
                "tensor_conversion": "author_ResizeDataset_float32_branch_ToTensor_no_divide255",
                "network": "author_InceptionV3W_resize_inside_False_layers_forward",
                "network_normalization": "(float32_pixels-128)/128_inside_author_wrapper",
                "generated_quantization_before_adapter": "numpy_rint((tanh_output+1)*127.5)_uint8",
            },
            "official_source_reference": "https://github.com/GaParmar/clean-fid",
            "KID_estimator": "caller_defined_unequal_count_U_statistic_not_library_subset_average",
        }

    def __call__(self, images):
        pixels = np.asarray(images)
        if (pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3
                or len(pixels) < 1 or min(pixels.shape[1:3]) < 1):
            raise ValueError("Clean-FID adapter needs a nonempty uint8 BHWC RGB batch")
        # Delegate per-channel float-PIL resizing to the author implementation.
        # Its result is float32, so ToTensor does NOT divide by255 (utils.py).
        tensors = [self.to_tensor(self.resize(image)) for image in pixels]
        batch = self.torch.stack(tensors).to(self.device)
        if tuple(batch.shape[1:]) != (3, 299, 299):
            raise ValueError("Author clean resizer returned an unexpected shape")
        with self.torch.inference_mode():
            result = self.model(batch)
        features = result.detach().float().cpu().numpy().astype(np.float64)
        if features.shape != (len(pixels), 2048) or not np.isfinite(features).all():
            raise FloatingPointError("Clean-FID returned nonfinite or incorrectly shaped features")
        return features
