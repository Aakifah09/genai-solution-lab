"""Static Matplotlib figures saved with each experiment (for reports and export)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: safe inside background jobs
import matplotlib.pyplot as plt  # noqa: E402

from src.persistence.storage import _replace_with_retry  # noqa: E402

_STYLE = {"figure.facecolor": "#1e2127", "axes.facecolor": "#1e2127", "axes.edgecolor": "#5c6370",
          "axes.labelcolor": "#c8ccd4", "text.color": "#c8ccd4", "xtick.color": "#9da5b4", "ytick.color": "#9da5b4",
          "grid.color": "#3e4451", "legend.frameon": False}
_C1, _C2 = "#61afef", "#e5c07b"


def plot_training_curves(history: list[dict[str, Any]], model: str, path: Path) -> Path:
    """Two-panel figure: GAN losses + D outputs, or VAE reconstruction + KL (train vs validation)."""
    epochs = [h["epoch"] for h in history]
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=120)
        if model == "gan":
            panels = [("BCE loss", [("d_loss", "Discriminator", _C1), ("g_loss", "Generator", _C2)]),
                      ("Mean D output", [("d_real_prob", "D(x) real", _C1), ("d_fake_prob", "D(G(z)) fake", _C2)])]
        else:
            panels = [("Reconstruction (nats/image)", [("train_recon", "train", _C1), ("val_recon", "validation", _C2)]),
                      ("KL divergence (nats/image)", [("train_kl", "train", _C1), ("val_kl", "validation", _C2)])]
        for ax, (ylabel, series) in zip(axes, panels):
            for key, label, color in series:
                ax.plot(epochs, [h[key] for h in history], marker="o", ms=3, lw=2, color=color, label=label)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend()
        fig.suptitle(f"{model.upper()} training curves", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.tmp.png")
        fig.savefig(tmp)
        plt.close(fig)
    _replace_with_retry(str(tmp), path)
    return path
