"""Fréchet Inception Distance (standard) and FD-MNIST (clearly labelled alternative).

Standard FID
------------
Uses ``pytorch-fid``'s port of the TensorFlow Inception-v3 graph (pool3, 2048-d),
the same weights used by the original FID paper, so values are computed with the
reference feature extractor. Grayscale 28x28 images are replicated to 3 channels and
bilinearly resized to 299x299 inside the network.

Limitations on MNIST (documented in docs/evaluation.md):
* Inception was trained on ImageNet photographs; MNIST digits are far outside that
  distribution, so features capture stroke texture more than digit identity.
* FID is biased upward for small sample sizes; values are only comparable between
  runs that use the same number of samples and the same reference set.
* Published MNIST FIDs differ in reference set, sample count and resizing, so the
  numbers here are for *within-project* comparison only.

FD-MNIST (non-standard)
-----------------------
Same Fréchet distance formula, but features come from the project's held-out MNIST
classifier (128-d penultimate layer). It is domain-appropriate but NOT standard FID
and is not comparable with any published number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation.metrics import feature_statistics, frechet_distance, frechet_distance_from_features

INCEPTION_DESCRIPTION = "pytorch-fid InceptionV3 (TF FID weights, pool3 2048-d), 1->3 channels, bilinear resize to 299x299"


class FIDUnavailableError(RuntimeError):
    """Raised when the Inception model cannot be loaded (missing package or weights download failed)."""


_INCEPTION_CACHE: dict[str, torch.nn.Module] = {}


def load_inception(device: torch.device) -> torch.nn.Module:
    key = str(device)
    if key in _INCEPTION_CACHE:
        return _INCEPTION_CACHE[key]
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError as exc:
        raise FIDUnavailableError("pytorch-fid is not installed (`pip install pytorch-fid`)") from exc
    try:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx], resize_input=True, normalize_input=True).to(device).eval()
    except Exception as exc:  # network/SSL/hash errors while fetching the FID weights
        raise FIDUnavailableError(
            f"Could not load FID Inception weights ({type(exc).__name__}: {exc}). They are downloaded "
            "once (~92 MB) from github.com/mseitzer/pytorch-fid releases into the torch hub cache."
        ) from exc
    _INCEPTION_CACHE[key] = model
    return model


@torch.no_grad()
def inception_features(images01: torch.Tensor, device: torch.device, batch_size: int = 100) -> np.ndarray:
    model = load_inception(device)
    feats = []
    for start in range(0, len(images01), batch_size):
        x = images01[start:start + batch_size].to(device).repeat(1, 3, 1, 1)
        feats.append(model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(feats).astype(np.float64)


def reference_statistics(real01: torch.Tensor, device: torch.device, cache_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Inception mean/covariance of the real reference set, cached on disk (it never changes)."""
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as cached:  # context manager releases the Windows file handle
            if int(cached["n"]) == len(real01):
                return cached["mu"], cached["sigma"]
    mu, sigma = feature_statistics(inception_features(real01, device))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.stem + ".tmp.npz")
        np.savez(tmp, mu=mu, sigma=sigma, n=len(real01))
        tmp.replace(cache_path)
    return mu, sigma


def compute_fid(gen01: torch.Tensor, real01: torch.Tensor, device: torch.device, cache_path: Path | None = None) -> dict[str, Any]:
    """Standard FID between generated images and a real reference set (both [0, 1], same count)."""
    if len(gen01) != len(real01):
        raise ValueError("FID sample counts must match for a fair comparison")
    real_mu, real_sigma = reference_statistics(real01, device, cache_path)
    gen_mu, gen_sigma = feature_statistics(inception_features(gen01, device))
    value = frechet_distance(gen_mu, gen_sigma, real_mu, real_sigma)
    return {
        "status": "computed",
        "value": value,
        "feature_extractor": INCEPTION_DESCRIPTION,
        "n_generated": len(gen01),
        "n_real": len(real01),
        "reference_set": "MNIST validation split",
    }


def compute_fd_mnist(gen_features: torch.Tensor, real_features: torch.Tensor) -> dict[str, Any]:
    return {
        "value": frechet_distance_from_features(gen_features.numpy(), real_features.numpy()),
        "feature_extractor": "project MNIST classifier, 128-d penultimate layer (NOT standard FID)",
        "n_generated": len(gen_features),
        "n_real": len(real_features),
        "reference_set": "MNIST validation split",
    }
