"""Pure (Streamlit-free) helpers used by the dashboard. Unit-tested in tests/test_ui_helpers.py."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.persistence.storage import COMPARABLE_METRICS

ARCHITECTURE_MERMAID = """flowchart TD
    A["Dataset layer<br/>torchvision MNIST download"] --> B["Validation and preprocessing<br/>uint8 checks, fixed-seed train/val split,<br/>per-model scaling"]
    B --> C["Experiment config and reproducibility<br/>typed ExperimentConfig, seeds, device detection"]
    C --> D1["GAN training<br/>DCGAN generator + discriminator"]
    C --> D2["VAE training<br/>conv encoder/decoder, ELBO"]
    D1 --> E["Checkpoint and experiment persistence<br/>atomic JSON/CSV, resumable last.pt"]
    D2 --> E
    E --> F["Evaluation pipeline<br/>held-out classifier, validity, diversity,<br/>FD-MNIST, Inception FID"]
    F --> G["Metrics, sample grids, comparison"]
    G --> H["Streamlit dashboard and CSV/JSON export"]
    H -. "launch / cancel / resume<br/>(background process)" .-> C
"""

METRIC_DEFINITIONS: list[dict[str, str]] = [
    {"metric": "GAN D loss / G loss", "kind": "Model objective (GAN only)",
     "meaning": "BCE losses of the adversarial game. D loss sums real and fake terms (2·ln2 ≈ 1.386 when D outputs 0.5 everywhere). They track training dynamics, not image quality.",
     "direction": "No 'better' direction; look for stability and no collapse."},
    {"metric": "D(x), D(G(z))", "kind": "Training diagnostic (GAN only)",
     "meaning": "Mean discriminator probability on real / generated batches. D(x)≈1 and D(G(z))≈0 for many epochs means the discriminator dominates.",
     "direction": "Diagnostic only. This is not a classification accuracy."},
    {"metric": "Reconstruction loss", "kind": "Model objective (VAE only)",
     "meaning": "Bernoulli negative log-likelihood: BCE summed over 784 pixels, averaged over images (nats/image). Reported on train and held-out validation data.",
     "direction": "Lower = more faithful reconstructions"},
    {"metric": "KL divergence", "kind": "Model objective (VAE only)",
     "meaning": "KL(q(z|x) || N(0,I)) summed over latent dims, averaged over images. Measures how far the encoder's posterior is from the prior.",
     "direction": "Trade-off: too low = posterior collapse, too high = poor sampling"},
    {"metric": "Validity rate", "kind": "Proxy (classifier-based)",
     "meaning": "Share of generated images whose top-class probability from the held-out MNIST classifier is ≥ threshold. Compare against the same rate on real validation images.",
     "direction": "Higher is better, capped by the real-data reference"},
    {"metric": "Class entropy (normalised)", "kind": "Proxy (classifier-based)",
     "meaning": "Entropy of the predicted-digit histogram ÷ ln 10. 1.0 means all digits are equally represented; low values indicate mode collapse.",
     "direction": "Closer to the real-data value (~1.0) is better"},
    {"metric": "Diversity ratio", "kind": "Proxy",
     "meaning": "Mean pairwise L2 distance among generated samples ÷ the same for real validation images (pixel space and classifier-feature space).",
     "direction": "Closer to 1 is better. Above 1 can mean noisy samples."},
    {"metric": "FD-MNIST", "kind": "Proxy distribution distance (non-standard)",
     "meaning": "Fréchet distance between Gaussians fitted to the classifier's 128-d features of generated and real validation images. It is not standard FID.",
     "direction": "Lower is better"},
    {"metric": "FID", "kind": "Distribution metric (standard)",
     "meaning": "Fréchet distance in Inception-v3 pool3 space (pytorch-fid TF weights) vs. real validation images. ImageNet features are a poor fit for digits, and the value is biased upward at small N.",
     "direction": "Lower is better. Compare only at equal sample counts."},
    {"metric": "Runtime", "kind": "Runtime",
     "meaning": "Wall-clock training and evaluation seconds on the recorded device.",
     "direction": "Context only. It depends on hardware and settings."},
]

STATE_ICONS = {
    "queued": ":material/schedule:", "running": ":material/play_circle:", "evaluating": ":material/analytics:",
    "completed": ":material/check_circle:", "failed": ":material/error:", "cancelled": ":material/cancel:",
    "interrupted": ":material/warning:", "unknown": ":material/help:",
}


def format_duration(seconds: float | None) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "-"
    if seconds < 0:
        raise ValueError("duration cannot be negative")
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_metric(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def experiment_label(row: dict[str, Any]) -> str:
    """'GAN · 2026-10-26 20:41 · 5 ep · 55,000 imgs · completed'"""
    exp_id = row.get("experiment_id", "?")
    parts = exp_id.split("-")
    when = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:]} {parts[2][:2]}:{parts[2][2:4]}" if len(parts) >= 3 else exp_id
    model = str(row.get("model", parts[0])).upper()
    bits = [model, when]
    if row.get("epochs") is not None and not pd.isna(row.get("epochs")):
        bits.append(f"{int(row['epochs'])} ep")
    if row.get("train_size") is not None and not pd.isna(row.get("train_size")):
        bits.append(f"{int(row['train_size']):,} imgs")
    bits.append(str(row.get("state", "unknown")))
    return " · ".join(bits)


def history_to_long(history: pd.DataFrame, columns: Iterable[str], labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Melt selected per-epoch columns into (epoch, series, value) rows for charting."""
    cols = [c for c in columns if c in history.columns]
    if history.empty or not cols:
        return pd.DataFrame(columns=["epoch", "series", "value"])
    long = history[["epoch", *cols]].melt(id_vars="epoch", var_name="series", value_name="value")
    if labels:
        long["series"] = long["series"].map(lambda s: labels.get(s, s))
    return long


def config_differences(configs: dict[str, dict[str, Any]], ignore: Iterable[str] = ("model",)) -> list[str]:
    """Config keys whose values differ between the given experiments."""
    if len(configs) < 2:
        return []
    keys = sorted(set().union(*[c.keys() for c in configs.values()]) - set(ignore))
    return [k for k in keys if len({repr(c.get(k)) for c in configs.values()}) > 1]


FAIRNESS_KEYS = ("epochs", "train_limit", "val_size", "eval_num_samples", "fid_num_samples", "validity_threshold")


def fairness_warnings(configs: dict[str, dict[str, Any]]) -> list[str]:
    """Settings that make metric comparisons unfair if they differ."""
    diffs = set(config_differences(configs))
    warnings = []
    for key in FAIRNESS_KEYS:
        if key in diffs:
            values = ", ".join(f"{eid.split('-')[0].upper()}…{eid[-6:]}={c.get(key)}" for eid, c in configs.items())
            warnings.append(f"`{key}` differs ({values})")
    return warnings


def comparison_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Table of metrics defined identically for every model, plus their 'better' direction."""
    records = []
    for name, (_, direction) in COMPARABLE_METRICS.items():
        rec: dict[str, Any] = {"metric": name, "better": direction}
        for row in rows:
            rec[row["experiment_id"]] = row.get(name)
        records.append(rec)
    return pd.DataFrame(records)


def interpret_comparison(rows: list[dict[str, Any]], labels: dict[str, str]) -> list[str]:
    """Plain statements derived only from the numbers; no claims beyond them."""
    out: list[str] = []
    if len(rows) < 2:
        return out
    for name, (_, direction) in COMPARABLE_METRICS.items():
        if direction not in ("higher", "lower", "closer to 1"):
            continue
        vals = [(r["experiment_id"], r.get(name)) for r in rows if isinstance(r.get(name), (int, float)) and not pd.isna(r.get(name))]
        if len(vals) < 2:
            continue
        if direction == "closer to 1":
            best = min(vals, key=lambda kv: abs(kv[1] - 1))
            worst = max(vals, key=lambda kv: abs(kv[1] - 1))
        elif direction == "higher":
            best, worst = max(vals, key=lambda kv: kv[1]), min(vals, key=lambda kv: kv[1])
        else:
            best, worst = min(vals, key=lambda kv: kv[1]), max(vals, key=lambda kv: kv[1])
        if best[1] == worst[1]:
            out.append(f"**{name}**: tied at {best[1]:.4g}.")
            continue
        out.append(f"**{name}** ({direction} is better): {labels.get(best[0], best[0])} = {best[1]:.4g} "
                   f"vs. {labels.get(worst[0], worst[0])} = {worst[1]:.4g}.")
    times = [(r["experiment_id"], r.get("train_seconds")) for r in rows if isinstance(r.get("train_seconds"), (int, float))]
    if len(times) >= 2:
        fastest = min(times, key=lambda kv: kv[1])
        out.append(f"**Training time**: fastest was {labels.get(fastest[0], fastest[0])} ({format_duration(fastest[1])}). "
                   "Time depends on epochs, data size and hardware, so it is not a quality measure.")
    return out


def parse_junit(path: Path) -> dict[str, Any] | None:
    """Summarise a pytest JUnit XML report (software tests, not model metrics)."""
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return None
    cases = []
    for case in suite.iter("testcase"):
        outcome = "passed"
        for tag in ("failure", "error", "skipped"):
            if case.find(tag) is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
        cases.append({"test": f"{case.get('classname')}::{case.get('name')}", "outcome": outcome, "seconds": float(case.get("time", 0))})
    tests = int(suite.get("tests", len(cases)))
    failures, errors, skipped = (int(suite.get(k, 0)) for k in ("failures", "errors", "skipped"))
    return {
        "tests": tests, "failures": failures, "errors": errors, "skipped": skipped,
        "passed": tests - failures - errors - skipped, "seconds": float(suite.get("time", 0)),
        "timestamp": suite.get("timestamp"), "cases": cases,
    }
