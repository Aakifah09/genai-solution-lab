import math

import numpy as np
import pytest
import torch

from src.evaluation.classifier import DigitClassifier, predict, train_classifier
from src.evaluation.fid import compute_fd_mnist
from src.evaluation.metrics import (
    diversity_metrics,
    feature_statistics,
    frechet_distance,
    mean_pairwise_distance,
    validity_metrics,
)


def one_hot_probs(classes, confidence):
    probs = torch.full((len(classes), 10), (1 - confidence) / 9)
    probs[torch.arange(len(classes)), torch.tensor(classes)] = confidence
    return probs


def test_validity_counts_threshold_inclusively():
    probs = torch.cat([one_hot_probs([0, 1], 0.95), one_hot_probs([2, 3], 0.5)])
    m = validity_metrics(probs, threshold=0.95)
    assert m["valid_fraction"] == pytest.approx(0.5)
    assert m["mean_max_confidence"] == pytest.approx(0.725)
    assert m["predicted_class_counts"][:4] == [1, 1, 1, 1]
    assert m["classes_covered"] == 4


def test_class_entropy_extremes():
    uniform = validity_metrics(one_hot_probs(list(range(10)) * 5, 0.99), 0.9)
    collapsed = validity_metrics(one_hot_probs([7] * 50, 0.99), 0.9)
    assert uniform["class_entropy_normalized"] == pytest.approx(1.0)
    assert collapsed["class_entropy_normalized"] == pytest.approx(0.0)
    assert collapsed["classes_covered"] == 1


def test_validity_rejects_bad_input():
    with pytest.raises(ValueError):
        validity_metrics(torch.rand(4, 5), 0.9)
    with pytest.raises(ValueError):
        validity_metrics(torch.rand(4, 10), 1.0)


def test_mean_pairwise_distance_known_values():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    # pairs: (0,1)=5, (0,2)=0, (1,2)=5 -> mean 10/3
    assert mean_pairwise_distance(x) == pytest.approx(10 / 3)
    assert mean_pairwise_distance(torch.ones(5, 3)) == pytest.approx(0.0)
    # Chunking must not change the result.
    y = torch.randn(50, 7)
    assert mean_pairwise_distance(y, chunk=7) == pytest.approx(mean_pairwise_distance(y, chunk=1000))


def test_diversity_ratio_is_one_for_identical_sets():
    imgs, feats = torch.rand(20, 1, 28, 28), torch.randn(20, 8)
    m = diversity_metrics(imgs, feats, imgs, feats)
    assert m["pixel_ratio_to_real"] == pytest.approx(1.0)
    assert m["feature_ratio_to_real"] == pytest.approx(1.0)


def test_frechet_distance_identical_is_zero_and_gaussian_closed_form():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(500, 4))
    mu, sigma = feature_statistics(feats)
    assert frechet_distance(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-6)
    # Diagonal covariances: FD = |mu1-mu2|^2 + sum (s1 + s2 - 2 sqrt(s1 s2))
    mu1, mu2 = np.array([0.0, 1.0]), np.array([1.0, 1.0])
    s1, s2 = np.diag([1.0, 4.0]), np.diag([4.0, 1.0])
    expected = 1.0 + (1 + 4 - 2 * 2) + (4 + 1 - 2 * 2)
    assert frechet_distance(mu1, s1, mu2, s2) == pytest.approx(expected)


def test_frechet_distance_grows_with_mean_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(400, 3))
    near = compute_fd_mnist(torch.tensor(base + 0.1), torch.tensor(base))["value"]
    far = compute_fd_mnist(torch.tensor(base + 2.0), torch.tensor(base))["value"]
    assert 0 < near < far
    assert far == pytest.approx(3 * 4.0, rel=0.01)


def test_classifier_shapes_and_probabilities():
    model = DigitClassifier()
    probs, feats = predict(model, torch.rand(7, 1, 28, 28), torch.device("cpu"), batch_size=3)
    assert probs.shape == (7, 10) and feats.shape == (7, 128)
    assert torch.allclose(probs.sum(1), torch.ones(7), atol=1e-5)


def test_classifier_learns_separable_synthetic_data():
    """Two trivially separable 'digits' (dark vs bright images); checks the training loop works."""
    g = torch.Generator().manual_seed(0)
    dark = torch.randint(0, 40, (200, 28, 28), generator=g, dtype=torch.uint8)
    bright = torch.randint(200, 256, (200, 28, 28), generator=g, dtype=torch.uint8)
    x = torch.cat([dark, bright])
    y = torch.cat([torch.zeros(200), torch.ones(200)]).long()
    perm = torch.randperm(400, generator=g)
    x, y = x[perm], y[perm]
    model, report = train_classifier(x[:300], y[:300], x[300:], y[300:], torch.device("cpu"), epochs=2, batch_size=32, log=lambda _: None)
    assert report["best_val_accuracy"] >= 0.95
    assert len(report["history"]) == 2
    assert math.isfinite(report["history"][-1]["train_loss"])


def test_held_out_test_report_uses_sklearn_metrics(monkeypatch, synthetic_mnist):
    from src.evaluation import classifier as clf_module

    images, labels = synthetic_mnist
    monkeypatch.setattr(clf_module, "load_test_split", lambda: (images[:100], labels[:100]))
    report = clf_module.held_out_test_report(DigitClassifier().eval(), torch.device("cpu"))
    assert report["test_size"] == 100
    assert 0 <= report["test_accuracy"] <= 1 and 0 <= report["test_macro_f1"] <= 1
    assert set(report["test_per_class"]) == {str(d) for d in range(10)}
    cm = np.array(report["test_confusion_matrix"])
    assert cm.shape == (10, 10) and cm.sum() == 100
    assert report["test_accuracy"] == pytest.approx(np.trace(cm) / 100)


@pytest.mark.parametrize("model, history", [
    ("gan", [{"epoch": e, "d_loss": 1.0, "g_loss": 2.0, "d_real_prob": 0.7, "d_fake_prob": 0.3} for e in (1, 2)]),
    ("vae", [{"epoch": e, "train_recon": 100.0, "val_recon": 101.0, "train_kl": 20.0, "val_kl": 20.5} for e in (1, 2)]),
])
def test_training_curve_figure_is_written(tmp_path, model, history):
    from PIL import Image

    from src.evaluation.plots import plot_training_curves

    path = plot_training_curves(history, model, tmp_path / "curves.png")
    assert path.exists() and Image.open(path).size[0] > 500
    assert list(tmp_path.glob(".*tmp*")) == []
