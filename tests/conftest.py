"""Shared fixtures. Unit tests use synthetic tensors only; nothing here downloads MNIST."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Redirect data/output dirs so tests never touch real experiments or download data."""
    monkeypatch.setenv("GENAI_LAB_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("GENAI_LAB_DATA_DIR", str(tmp_path / "data"))
    yield tmp_path


@pytest.fixture
def synthetic_mnist():
    """600 fake 28x28 uint8 'digits' with balanced labels (deterministic)."""
    g = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (600, 28, 28), generator=g, dtype=torch.uint8)
    labels = torch.arange(600) % 10
    return images, labels


@pytest.fixture
def store(tmp_path):
    from src.persistence.storage import ExperimentStore

    return ExperimentStore(tmp_path / "outputs" / "experiments")
