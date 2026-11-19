"""Restricted local-tensor loading; no network access or dataset download."""
import hashlib
from pathlib import Path
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_tensor(path, unconditional=False, max_bytes=512 * 1024 ** 2):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > max_bytes:
        raise ValueError("Expected a regular local tensor file within the 512MiB reference cap")
    before = sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, torch.Tensor):
        x, labels = payload, None
    elif isinstance(payload, dict) and set(payload) <= {"x", "labels"} and "x" in payload:
        x, labels = payload["x"], payload.get("labels")
    else:
        raise ValueError("Expected a tensor or {'x': tensor, 'labels': int64 tensor}")
    if not isinstance(x, torch.Tensor) or x.ndim not in (2, 3, 4) or len(x) < 2:
        raise ValueError("Expected N>=2 observations with 1–3 feature/image axes")
    if x.numel() > 100_000_000:
        raise ValueError("Reference tensor exceeds 100 million elements")
    shape = list(x.shape[1:])
    if x.dtype == torch.uint8:
        x = x.float() / 127.5 - 1
    elif x.is_floating_point():
        x = x.float()
    else:
        raise ValueError("Use uint8 observations or floating observations already in [-1,1]")
    if not bool(torch.isfinite(x).all()) or float(x.abs().max()) > 1.00001:
        raise ValueError("Observations must be finite and lie in [-1,1]; no implicit clipping")
    if unconditional:
        labels = None
    classes = 0
    if labels is not None:
        if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.shape != (len(x),):
            raise ValueError("labels must be an N-vector of int64 class indices")
        if int(labels.min()) < 0:
            raise ValueError("Negative labels")
        classes = int(labels.max()) + 1
        if classes > 10_000 or not torch.equal(labels.unique(sorted=True), torch.arange(classes)):
            raise ValueError("Observed labels must cover the contiguous range 0,...,K-1")
    if sha256(path) != before:
        raise RuntimeError("Input changed during reading")
    return x.reshape(len(x), -1).contiguous(), labels, shape, classes, before


def draw_batch(x, labels, batch_size, balanced=False):
    if balanced:
        if labels is None:
            raise ValueError("Balanced sampling requires labels")
        classes = int(labels.max()) + 1
        if batch_size < classes or batch_size % classes:
            raise ValueError("Balanced batch size must be a positive multiple of class count")
        parts = []
        for group in range(classes):
            eligible = (labels == group).nonzero(as_tuple=True)[0]
            parts.append(eligible[torch.randint(len(eligible), (batch_size // classes,))])
        index = torch.cat(parts)
        index = index[torch.randperm(len(index))]
    else:
        index = torch.randint(len(x), (batch_size,))
    return x[index], None if labels is None else labels[index]


def requested_labels(n, classes, class_label=None, device="cpu"):
    if not classes:
        if class_label is not None:
            raise ValueError("Unconditional model has no class label")
        return None
    if class_label is None:
        return torch.arange(n, device=device, dtype=torch.long) % classes
    if not 0 <= class_label < classes:
        raise ValueError("Requested class outside configured range")
    return torch.full((n,), class_label, device=device, dtype=torch.long)

