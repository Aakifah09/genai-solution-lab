"""MNIST loading, leakage-free splitting and batching.

Split policy
------------
* MNIST's official 60k "train" set is permuted with the fixed ``SPLIT_SEED`` and cut
  into a **training split** (60k - val_size) and a **validation split** (val_size).
* Generative models and the evaluator classifier train **only** on the training split.
* The validation split is the real-data reference for generative metrics (FID,
  FD-MNIST, diversity baselines) and for classifier model selection.
* The official 10k **test** set is never touched by generative training or metric
  tuning. It is used exactly once: to report the evaluator classifier's held-out
  accuracy.
* ``train_limit`` takes the first N indices of the (already shuffled) training split,
  so smaller experiments use a subset of larger ones and never reach validation data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocessing import Normalization, to_model_range, uint8_to_unit
from src.utils.config import MNIST_TRAIN_POOL, SPLIT_SEED, data_dir


class DatasetUnavailableError(RuntimeError):
    """MNIST could not be loaded or downloaded."""


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray

    def check_disjoint(self) -> None:
        overlap = np.intersect1d(self.train, self.val)
        if overlap.size:
            raise AssertionError(f"train/val overlap detected ({overlap.size} indices) - data leakage")


def make_train_val_split(n_total: int, val_size: int, seed: int = SPLIT_SEED) -> SplitIndices:
    """Deterministically split ``range(n_total)`` into disjoint train/val index arrays."""
    if not 0 < val_size < n_total:
        raise ValueError(f"val_size must be in (0, {n_total}), got {val_size}")
    perm = np.random.default_rng(seed).permutation(n_total)
    split = SplitIndices(train=perm[val_size:], val=perm[:val_size])
    split.check_disjoint()
    return split


def limit_indices(indices: np.ndarray, limit: int | None) -> np.ndarray:
    """Take the first ``limit`` indices (the split is already randomly permuted)."""
    if limit is None:
        return indices
    if limit <= 0 or limit > len(indices):
        raise ValueError(f"limit must be in [1, {len(indices)}], got {limit}")
    return indices[:limit]


def mnist_available(root: Path | None = None) -> bool:
    raw = (root or data_dir()) / "MNIST" / "raw"
    needed = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    return all((raw / name).exists() for name in needed)


def load_mnist_raw(train: bool, root: Path | None = None, download: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (uint8 images (N, 28, 28), int64 labels (N,)) for the official train or test set."""
    from torchvision.datasets import MNIST

    root = root or data_dir()
    try:
        ds = MNIST(root=str(root), train=train, download=download)
    except RuntimeError as exc:  # torchvision raises RuntimeError("Dataset not found...")
        raise DatasetUnavailableError(
            f"MNIST not found under {root} and download={download}. "
            "Run `python cli.py download-data` or enable downloading."
        ) from exc
    except OSError as exc:  # network / filesystem failures during download
        raise DatasetUnavailableError(
            f"Could not download MNIST into {root}: {exc}. Check your internet connection or proxy, "
            "or place the raw MNIST files in data/MNIST/raw manually."
        ) from exc
    images, labels = ds.data, ds.targets
    if images.dtype != torch.uint8 or tuple(images.shape[1:]) != (28, 28):
        raise DatasetUnavailableError(f"Unexpected MNIST tensor format: {images.dtype} {tuple(images.shape)}")
    return images, labels.long()


@dataclass
class MNISTSplits:
    """uint8 images + labels for the training and validation splits (no test data)."""

    train_images: torch.Tensor
    train_labels: torch.Tensor
    val_images: torch.Tensor
    val_labels: torch.Tensor

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train_labels), "val": len(self.val_labels)}


def split_tensors(
    images: torch.Tensor, labels: torch.Tensor, val_size: int, train_limit: int | None = None, seed: int = SPLIT_SEED
) -> MNISTSplits:
    """Apply the project split policy to any (images, labels) pair (used with real MNIST and in tests)."""
    if len(images) != len(labels):
        raise ValueError(f"images ({len(images)}) and labels ({len(labels)}) differ in length")
    split = make_train_val_split(len(images), val_size, seed)
    train_idx = torch.from_numpy(limit_indices(split.train, train_limit))
    val_idx = torch.from_numpy(split.val)
    return MNISTSplits(images[train_idx], labels[train_idx], images[val_idx], labels[val_idx])


def load_splits(val_size: int, train_limit: int | None = None, root: Path | None = None, download: bool = True) -> MNISTSplits:
    images, labels = load_mnist_raw(train=True, root=root, download=download)
    if len(images) != MNIST_TRAIN_POOL:
        raise DatasetUnavailableError(f"Expected {MNIST_TRAIN_POOL} MNIST training images, found {len(images)}")
    return split_tensors(images, labels, val_size, train_limit)


def load_test_split(root: Path | None = None, download: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Official MNIST test set. Only used for the evaluator classifier's final held-out accuracy."""
    return load_mnist_raw(train=False, root=root, download=download)


class _NormalizeCollate:
    """Collate uint8 samples into a float batch in the model's range (picklable for workers)."""

    def __init__(self, normalization: Normalization):
        self.normalization = normalization

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return to_model_range(uint8_to_unit(images), self.normalization), labels


def make_loader(
    images: torch.Tensor,
    labels: torch.Tensor,
    normalization: Normalization,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader over uint8 tensors; conversion happens per batch to keep memory low.

    Shuffling uses a dedicated seeded generator so batch order is reproducible.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if drop_last and len(images) < batch_size:
        raise ValueError(f"dataset of {len(images)} images is smaller than batch_size {batch_size}")
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=gen,
        collate_fn=_NormalizeCollate(normalization),
        num_workers=0,  # tensors are in memory; worker processes would only add Windows spawn overhead
    )


def class_distribution(labels: torch.Tensor | np.ndarray, num_classes: int = 10) -> np.ndarray:
    arr = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    return np.bincount(arr.astype(np.int64), minlength=num_classes)
