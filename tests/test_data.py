import numpy as np
import pytest
import torch

from src.data.dataset import (
    DatasetUnavailableError,
    class_distribution,
    limit_indices,
    load_mnist_raw,
    make_loader,
    make_train_val_split,
    split_tensors,
)
from src.data.preprocessing import (
    DataValidationError,
    normalization_for_model,
    to_model_range,
    to_unit_range,
    uint8_to_unit,
    validate_unit_images,
)


def test_uint8_to_unit_scales_and_adds_channel():
    x = torch.tensor([[[0, 255] * 14] * 28], dtype=torch.uint8)
    out = uint8_to_unit(x)
    assert out.shape == (1, 1, 28, 28)
    assert out.min() == 0.0 and out.max() == 1.0


@pytest.mark.parametrize("bad", [torch.rand(2, 28, 28), torch.zeros(2, 3, 28, 28, dtype=torch.uint8), torch.zeros(2, 32, 32, dtype=torch.uint8)])
def test_uint8_to_unit_rejects_bad_input(bad):
    with pytest.raises(DataValidationError):
        uint8_to_unit(bad)


def test_model_ranges_round_trip():
    x = torch.rand(4, 1, 28, 28)
    assert normalization_for_model("gan") == "tanh" and normalization_for_model("vae") == "unit"
    tanh = to_model_range(x, "tanh")
    assert tanh.min() >= -1 and tanh.max() <= 1
    assert torch.allclose(to_unit_range(tanh, "tanh"), x, atol=1e-6)
    assert torch.equal(to_model_range(x, "unit"), x)


def test_validate_unit_images_catches_problems():
    validate_unit_images(torch.rand(2, 1, 28, 28))
    with pytest.raises(DataValidationError, match="NaN"):
        validate_unit_images(torch.full((1, 1, 28, 28), float("nan")))
    with pytest.raises(DataValidationError, match=r"\[0, 1\]"):
        validate_unit_images(torch.full((1, 1, 28, 28), 2.0))
    with pytest.raises(DataValidationError, match="shaped"):
        validate_unit_images(torch.rand(1, 28, 28))


def test_split_is_disjoint_complete_and_deterministic():
    a = make_train_val_split(1000, 100, seed=7)
    b = make_train_val_split(1000, 100, seed=7)
    assert len(a.val) == 100 and len(a.train) == 900
    assert np.intersect1d(a.train, a.val).size == 0
    assert np.array_equal(np.sort(np.concatenate([a.train, a.val])), np.arange(1000))
    assert np.array_equal(a.train, b.train) and np.array_equal(a.val, b.val)
    assert not np.array_equal(a.val, make_train_val_split(1000, 100, seed=8).val)


def test_limited_train_never_overlaps_validation_and_is_nested():
    split = make_train_val_split(1000, 200)
    small, large = limit_indices(split.train, 50), limit_indices(split.train, 500)
    assert np.intersect1d(small, split.val).size == 0
    assert np.isin(small, large).all()  # smaller experiments use a subset of larger ones
    with pytest.raises(ValueError):
        limit_indices(split.train, 801)


def test_split_tensors_isolates_images(synthetic_mnist):
    images, labels = synthetic_mnist
    # Make every image unique so overlap can be detected by content.
    images = images.clone()
    images[:, 0, 0] = torch.arange(600) % 256
    images[:, 0, 1] = torch.arange(600) // 256
    s = split_tensors(images, labels, val_size=100, train_limit=300)
    assert s.sizes == {"train": 300, "val": 100}
    train_keys = {tuple(x[0, :2].tolist()) for x in s.train_images}
    val_keys = {tuple(x[0, :2].tolist()) for x in s.val_images}
    assert train_keys.isdisjoint(val_keys)


def test_loader_batches_normalized_and_reproducible(synthetic_mnist):
    images, labels = synthetic_mnist
    l1 = make_loader(images, labels, "tanh", batch_size=64, shuffle=True, seed=3, drop_last=True)
    l2 = make_loader(images, labels, "tanh", batch_size=64, shuffle=True, seed=3, drop_last=True)
    x1, y1 = next(iter(l1))
    x2, y2 = next(iter(l2))
    assert x1.shape == (64, 1, 28, 28) and x1.dtype == torch.float32
    assert x1.min() >= -1 and x1.max() <= 1
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert len(l1) == 600 // 64


def test_loader_rejects_batch_larger_than_dataset_with_drop_last(synthetic_mnist):
    images, labels = synthetic_mnist
    with pytest.raises(ValueError, match="smaller than batch_size"):
        make_loader(images[:10], labels[:10], "unit", batch_size=64, shuffle=True, seed=0, drop_last=True)


def test_class_distribution_counts():
    assert class_distribution(torch.tensor([0, 1, 1, 9])).tolist() == [1, 2, 0, 0, 0, 0, 0, 0, 0, 1]


def test_missing_dataset_gives_actionable_error(tmp_path):
    with pytest.raises(DatasetUnavailableError, match="download-data"):
        load_mnist_raw(train=True, root=tmp_path / "empty", download=False)
