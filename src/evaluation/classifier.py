"""Held-out digit classifier used as an *evaluator* of generated images.

It is trained only on the training split, model-selected on the validation split,
and its accuracy is reported once on the official MNIST test set. That accuracy
describes the evaluator itself - it is NOT an accuracy of the GAN or VAE.

The 128-d penultimate layer doubles as an MNIST-specific feature extractor for the
FD-MNIST metric (see ``fid.py``).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from src.data.dataset import load_splits, load_test_split, make_loader
from src.data.preprocessing import MNIST_MEAN, MNIST_STD, uint8_to_unit
from src.persistence.storage import atomic_write_json, read_json, utc_now
from src.utils.config import SPLIT_SEED, evaluator_dir
from src.utils.reproducibility import set_seed

CLASSIFIER_FILE = "digit_classifier.pt"
REPORT_FILE = "classifier_report.json"
FEATURE_DIM = 128


class DigitClassifier(nn.Module):
    """Small CNN. Input: (N, 1, 28, 28) in [0, 1]; standardisation happens inside."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, FEATURE_DIM), nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(FEATURE_DIM, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv((x - MNIST_MEAN) / MNIST_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.features(x)))


@torch.no_grad()
def predict(model: DigitClassifier, images01: torch.Tensor, device: torch.device, batch_size: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (softmax probabilities (N, 10), penultimate features (N, 128)) on CPU."""
    model.eval()
    probs, feats = [], []
    for start in range(0, len(images01), batch_size):
        x = images01[start:start + batch_size].to(device)
        f = model.features(x)
        probs.append(F.softmax(model.head(f), dim=1).cpu())
        feats.append(f.cpu())
    return torch.cat(probs), torch.cat(feats)


def accuracy(model: DigitClassifier, images_u8: torch.Tensor, labels: torch.Tensor, device: torch.device) -> float:
    probs, _ = predict(model, uint8_to_unit(images_u8), device)
    return float((probs.argmax(1) == labels).float().mean())


def train_classifier(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[DigitClassifier, dict[str, Any]]:
    """Train on the training split; keep the epoch with the best validation accuracy."""
    set_seed(seed)
    model = DigitClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = make_loader(train_images, train_labels, "unit", batch_size, shuffle=True, seed=seed)
    best_acc, best_state, history = -1.0, None, []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        val_acc = accuracy(model, val_images, val_labels, device)
        history.append({"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total, "val_accuracy": val_acc})
        log(f"[classifier] epoch {epoch}/{epochs} loss={loss_sum / total:.4f} train_acc={correct / total:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    report = {
        "train_size": len(train_labels),
        "val_size": len(val_labels),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "best_val_accuracy": best_acc,
        "history": history,
        "train_seconds": time.perf_counter() - start,
        "device": device.type,
    }
    return model, report


def held_out_test_report(model: DigitClassifier, device: torch.device) -> dict[str, Any]:
    """One-time evaluation of the evaluator on the official MNIST test set (scikit-learn metrics)."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    test_images, test_labels = load_test_split()
    probs, _ = predict(model, uint8_to_unit(test_images), device)
    y_true, y_pred = test_labels.numpy(), probs.argmax(1).numpy()
    per_class = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_size": int(len(y_true)),
        "test_per_class": {str(d): {k: float(per_class[str(d)][k]) for k in ("precision", "recall", "f1-score")} for d in range(10)},
        "test_macro_f1": float(per_class["macro avg"]["f1-score"]),
        "test_confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(10))).tolist(),
    }


def classifier_paths(root: Path | None = None) -> tuple[Path, Path]:
    root = root or evaluator_dir()
    return root / CLASSIFIER_FILE, root / REPORT_FILE


def save_classifier(model: DigitClassifier, report: dict[str, Any], root: Path | None = None) -> None:
    weights, report_path = classifier_paths(root)
    weights.parent.mkdir(parents=True, exist_ok=True)
    tmp = weights.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    tmp.replace(weights)
    atomic_write_json(report_path, report)


def load_classifier(device: torch.device, root: Path | None = None) -> tuple[DigitClassifier, dict[str, Any]] | None:
    weights, report_path = classifier_paths(root)
    if not weights.exists() or not report_path.exists():
        return None
    model = DigitClassifier()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.to(device).eval(), read_json(report_path)


def get_or_train_classifier(
    device: torch.device, val_size: int = 5_000, force: bool = False, log: Callable[[str], None] = print, root: Path | None = None
) -> tuple[DigitClassifier, dict[str, Any]]:
    """Load the cached evaluator, or train it (train split), select it (val split) and test it once."""
    if not force:
        loaded = load_classifier(device, root)
        if loaded is not None:
            model, report = loaded
            if report.get("val_size") == val_size and report.get("split_seed") == SPLIT_SEED:
                if "test_per_class" not in report:  # older cache: add the sklearn report once
                    report.update(held_out_test_report(model, device))
                    save_classifier(model, report, root)
                return model, report
            log("[classifier] cached evaluator used a different split; retraining")
    log("[classifier] training held-out evaluator classifier (training split only)")
    splits = load_splits(val_size=val_size)
    model, report = train_classifier(splits.train_images, splits.train_labels, splits.val_images, splits.val_labels, device, log=log)
    report.update(held_out_test_report(model, device))
    report["split_seed"] = SPLIT_SEED
    report["created_at"] = utc_now()
    log(f"[classifier] held-out MNIST test accuracy: {report['test_accuracy']:.4f}")
    save_classifier(model, report, root)
    return model.eval(), report
