"""Image preprocessing and validation.

Raw MNIST pixels are uint8 in [0, 255]. Each consumer needs a different range:

* GAN   -> [-1, 1]  ("tanh"): matches the generator's Tanh output layer.
* VAE   -> [0, 1]   ("unit"): pixels are treated as Bernoulli probabilities for the
                              binary cross-entropy reconstruction term.
* Digit classifier / metrics -> [0, 1]; the classifier standardises internally.

Every evaluation path converts generated images back to [0, 1] first, so metrics
are computed on identical scales for both models.
"""

from __future__ import annotations

from typing import Literal

import torch

Normalization = Literal["tanh", "unit"]
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
IMAGE_SHAPE = (1, 28, 28)


class DataValidationError(ValueError):
    """Raised when image tensors have the wrong dtype, shape or value range."""


def normalization_for_model(model: str) -> Normalization:
    if model == "gan":
        return "tanh"
    if model in ("vae", "classifier"):
        return "unit"
    raise ValueError(f"No normalization defined for model {model!r}")


def uint8_to_unit(images: torch.Tensor) -> torch.Tensor:
    """Convert uint8 images shaped (N, 28, 28) or (N, 1, 28, 28) to float (N, 1, 28, 28) in [0, 1]."""
    if images.dtype != torch.uint8:
        raise DataValidationError(f"expected uint8 images, got {images.dtype}")
    if images.dim() == 3:
        images = images.unsqueeze(1)
    if images.dim() != 4 or tuple(images.shape[1:]) != IMAGE_SHAPE:
        raise DataValidationError(f"expected images shaped (N, 28, 28) or (N, 1, 28, 28), got {tuple(images.shape)}")
    return images.float().div_(255.0)


def to_model_range(x_unit: torch.Tensor, normalization: Normalization) -> torch.Tensor:
    if normalization == "unit":
        return x_unit
    if normalization == "tanh":
        return x_unit * 2.0 - 1.0
    raise ValueError(f"unknown normalization {normalization!r}")


def to_unit_range(x: torch.Tensor, normalization: Normalization) -> torch.Tensor:
    """Map model-space images back to [0, 1] (clamped) for display and evaluation."""
    if normalization == "unit":
        return x.clamp(0.0, 1.0)
    if normalization == "tanh":
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    raise ValueError(f"unknown normalization {normalization!r}")


def validate_unit_images(x: torch.Tensor, name: str = "images") -> None:
    """Check a float batch is (N, 1, 28, 28), finite, and inside [0, 1]."""
    if not torch.is_floating_point(x):
        raise DataValidationError(f"{name} must be a float tensor, got {x.dtype}")
    if x.dim() != 4 or tuple(x.shape[1:]) != IMAGE_SHAPE:
        raise DataValidationError(f"{name} must be shaped (N, 1, 28, 28), got {tuple(x.shape)}")
    if x.shape[0] == 0:
        raise DataValidationError(f"{name} is empty")
    if not torch.isfinite(x).all():
        raise DataValidationError(f"{name} contains NaN or Inf values")
    lo, hi = float(x.min()), float(x.max())
    if lo < -1e-6 or hi > 1 + 1e-6:
        raise DataValidationError(f"{name} must be in [0, 1], observed range [{lo:.4f}, {hi:.4f}]")


def preprocessing_summary(model: str) -> dict[str, str]:
    """Human-readable description used by the dashboard and saved metadata."""
    norm = normalization_for_model(model)
    return {
        "input": "MNIST uint8 grayscale, 1x28x28",
        "scaling": "x / 255 -> [0, 1]",
        "model_range": "[-1, 1] via 2x - 1 (matches Tanh output)" if norm == "tanh" else "[0, 1] (Bernoulli pixel probabilities)",
        "augmentation": "none",
        "evaluation_range": "[0, 1] for all models",
    }
