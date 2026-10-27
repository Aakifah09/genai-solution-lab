"""Metric calculations. Pure functions over tensors/arrays so they are unit-testable.

Proxy metrics (all computed with the held-out digit classifier):

* **Validity rate** - share of images whose top softmax probability is >= threshold.
  Measures "looks like *some* digit to a classifier trained on real MNIST", not
  correctness: a generator has no ground-truth label to be correct about.
* **Class entropy (normalised)** - entropy of the predicted-class histogram divided by
  ln(10). 1.0 = all ten digits equally represented; low values suggest mode collapse.
* **Diversity** - mean pairwise L2 distance between samples, in pixel space and in
  classifier-feature space, reported alongside the same statistic for an equally sized
  set of real validation images. Ratio ~1 means "as spread out as real data".
* **Fréchet distance** - distance between Gaussians fitted to feature sets; used by FID
  (Inception features) and FD-MNIST (classifier features).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy import linalg


def validity_metrics(probs: torch.Tensor, threshold: float) -> dict[str, Any]:
    if probs.dim() != 2 or probs.shape[1] != 10:
        raise ValueError(f"expected probabilities shaped (N, 10), got {tuple(probs.shape)}")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    max_prob, pred = probs.max(dim=1)
    counts = torch.bincount(pred, minlength=10).float()
    share = counts / counts.sum()
    nonzero = share[share > 0]
    entropy = float(-(nonzero * nonzero.log()).sum())
    return {
        "threshold": threshold,
        "valid_fraction": float((max_prob >= threshold).float().mean()),
        "mean_max_confidence": float(max_prob.mean()),
        "predicted_class_counts": [int(c) for c in counts],
        "class_entropy_normalized": entropy / math.log(10),
        "classes_covered": int((counts > 0).sum()),
    }


def mean_pairwise_distance(x: torch.Tensor, chunk: int = 1024) -> float:
    """Mean Euclidean distance over all unordered pairs of rows (exact, chunked)."""
    x = x.flatten(1).double()
    n = len(x)
    if n < 2:
        raise ValueError("need at least two samples to measure diversity")
    total = 0.0
    for start in range(0, n, chunk):
        total += float(torch.cdist(x[start:start + chunk], x).sum())
    return total / (n * (n - 1))  # diagonal distances are zero; each pair counted twice


def diversity_metrics(gen_images: torch.Tensor, gen_features: torch.Tensor, real_images: torch.Tensor, real_features: torch.Tensor) -> dict[str, Any]:
    gen_pix, real_pix = mean_pairwise_distance(gen_images), mean_pairwise_distance(real_images)
    gen_feat, real_feat = mean_pairwise_distance(gen_features), mean_pairwise_distance(real_features)
    return {
        "pixel_mean_pairwise_l2": gen_pix,
        "real_pixel_mean_pairwise_l2": real_pix,
        "pixel_ratio_to_real": gen_pix / real_pix,
        "feature_mean_pairwise_l2": gen_feat,
        "real_feature_mean_pairwise_l2": real_feat,
        "feature_ratio_to_real": gen_feat / real_feat,
        "n_samples": len(gen_images),
    }


def feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("features must be a 2-D array with at least two rows")
    return features.mean(axis=0), np.cov(features, rowvar=False)


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^{1/2}).

    Mirrors the reference FID implementation: if the matrix square root is singular,
    a small multiple of the identity is added; small imaginary parts from numerical
    error are discarded, large ones raise.
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError("mean / covariance shapes differ")
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary component too large: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def frechet_distance_from_features(gen: np.ndarray, real: np.ndarray) -> float:
    return frechet_distance(*feature_statistics(gen), *feature_statistics(real))
