"""Experiment persistence: one self-contained directory per experiment.

Layout of ``outputs/experiments/<experiment_id>/``::

    config.json      ExperimentConfig (written once at creation)
    job.json         launcher-owned: pid, command, launch time
    status.json      job-owned: state, progress, current losses, errors
    history.csv      per-epoch training metrics
    metrics.json     evaluation results (nested, with descriptions)
    metrics.csv      the same results, flat and downloadable
    metadata.json    config + software/device info + runtimes
    train.log        full log of the training process
    cancel.flag      present when cancellation was requested from the UI
    samples/         epoch_###.png, recon_epoch_###.png, eval_samples.png, ...
    checkpoints/     last.pt (resumable: model + optimizer + history)

Concurrency model: each file has exactly one writer (UI writes ``job.json`` /
``cancel.flag``; the training process writes everything else) and JSON/CSV writes
are atomic (temp file + ``os.replace``), so readers never observe partial files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import ExperimentConfig, experiments_dir

EXPERIMENT_ID_RE = re.compile(r"^(gan|vae)-\d{8}-\d{6}-[0-9a-f]{6}$")
ACTIVE_STATES = ("queued", "running", "evaluating")
TERMINAL_STATES = ("completed", "failed", "cancelled", "interrupted")


class StorageError(RuntimeError):
    pass


class ExperimentNotFoundError(StorageError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_experiment_id(model: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_id = f"{model}-{stamp}-{uuid.uuid4().hex[:6]}"
    validate_experiment_id(exp_id)
    return exp_id


def validate_experiment_id(exp_id: str) -> str:
    """Reject anything that is not a generated ID (prevents path traversal via IDs)."""
    if not isinstance(exp_id, str) or not EXPERIMENT_ID_RE.match(exp_id):
        raise StorageError(f"invalid experiment id {exp_id!r}")
    return exp_id


def _replace_with_retry(src: str, dst: Path, attempts: int = 20) -> None:
    # On Windows os.replace fails with PermissionError while another process holds
    # the destination open (e.g. the dashboard reading status.json); retry briefly.
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False, default=_json_default))


def _json_default(obj: Any) -> Any:
    # numpy scalars / paths sneak into metric dicts; convert rather than crash.
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    for attempt in range(10):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            time.sleep(0.05)
        except json.JSONDecodeError as exc:
            raise StorageError(f"{path} is corrupted ({exc}). Delete or repair the file.") from exc
    raise StorageError(f"could not read {path}: file is locked by another process")


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path

    @property
    def config(self) -> Path: return self.root / "config.json"
    @property
    def job(self) -> Path: return self.root / "job.json"
    @property
    def status(self) -> Path: return self.root / "status.json"
    @property
    def history(self) -> Path: return self.root / "history.csv"
    @property
    def metrics_json(self) -> Path: return self.root / "metrics.json"
    @property
    def metrics_csv(self) -> Path: return self.root / "metrics.csv"
    @property
    def metadata(self) -> Path: return self.root / "metadata.json"
    @property
    def log(self) -> Path: return self.root / "train.log"
    @property
    def stdout_log(self) -> Path: return self.root / "job_stdout.log"
    @property
    def cancel_flag(self) -> Path: return self.root / "cancel.flag"
    @property
    def samples(self) -> Path: return self.root / "samples"
    @property
    def checkpoints(self) -> Path: return self.root / "checkpoints"
    @property
    def checkpoint(self) -> Path: return self.checkpoints / "last.pt"

    def epoch_grid(self, epoch: int) -> Path:
        return self.samples / f"epoch_{epoch:03d}.png"

    def recon_grid(self, epoch: int) -> Path:
        return self.samples / f"recon_epoch_{epoch:03d}.png"


class ExperimentStore:
    """Create, list, read and update experiments under a root directory."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else experiments_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- creation / lookup -------------------------------------------------
    def create(self, config: ExperimentConfig) -> str:
        config.validate()
        exp_id = new_experiment_id(config.model)
        paths = self.paths(exp_id, must_exist=False)
        paths.root.mkdir(parents=True, exist_ok=False)
        paths.samples.mkdir()
        paths.checkpoints.mkdir()
        atomic_write_json(paths.config, config.to_dict())
        atomic_write_json(paths.status, {"state": "queued", "created_at": utc_now(), "updated_at": utc_now()})
        return exp_id

    def paths(self, exp_id: str, must_exist: bool = True) -> ExperimentPaths:
        validate_experiment_id(exp_id)
        paths = ExperimentPaths(self.root / exp_id)
        if must_exist and not paths.root.is_dir():
            raise ExperimentNotFoundError(f"experiment {exp_id} not found in {self.root}")
        return paths

    def exists(self, exp_id: str) -> bool:
        try:
            return self.paths(exp_id).root.is_dir()
        except StorageError:
            return False

    def experiment_ids(self) -> list[str]:
        ids = [p.name for p in self.root.iterdir() if p.is_dir() and EXPERIMENT_ID_RE.match(p.name)]
        return sorted(ids, key=lambda i: i.split("-", 1)[1], reverse=True)  # newest first

    def delete(self, exp_id: str) -> None:
        paths = self.paths(exp_id)
        if self.read_status(exp_id).get("state") in ACTIVE_STATES:
            raise StorageError(f"experiment {exp_id} is still active; cancel it before deleting")
        shutil.rmtree(paths.root)

    # -- config / status ---------------------------------------------------
    def load_config(self, exp_id: str) -> ExperimentConfig:
        data = read_json(self.paths(exp_id).config)
        if data is None:
            raise StorageError(f"experiment {exp_id} has no config.json")
        return ExperimentConfig.from_dict(data)

    def read_status(self, exp_id: str) -> dict[str, Any]:
        return read_json(self.paths(exp_id).status, default={"state": "unknown"})

    def update_status(self, exp_id: str, **fields: Any) -> dict[str, Any]:
        """Merge fields into status.json. Only the experiment's own process should call this
        while a job is alive (single-writer rule)."""
        status = self.read_status(exp_id)
        status.update(fields)
        status["updated_at"] = utc_now()
        atomic_write_json(self.paths(exp_id).status, status)
        return status

    def read_job(self, exp_id: str) -> dict[str, Any] | None:
        return read_json(self.paths(exp_id).job)

    def write_job(self, exp_id: str, job: dict[str, Any]) -> None:
        atomic_write_json(self.paths(exp_id).job, job)

    # -- cancellation ------------------------------------------------------
    def request_cancel(self, exp_id: str) -> None:
        atomic_write_text(self.paths(exp_id).cancel_flag, utc_now())

    def cancel_requested(self, exp_id: str) -> bool:
        return self.paths(exp_id).cancel_flag.exists()

    def clear_cancel(self, exp_id: str) -> None:
        self.paths(exp_id).cancel_flag.unlink(missing_ok=True)

    # -- results -----------------------------------------------------------
    def write_history(self, exp_id: str, history: list[dict[str, Any]]) -> None:
        atomic_write_text(self.paths(exp_id).history, pd.DataFrame(history).to_csv(index=False))

    def read_history(self, exp_id: str) -> pd.DataFrame:
        path = self.paths(exp_id).history
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def write_metrics(self, exp_id: str, metrics: dict[str, Any]) -> None:
        paths = self.paths(exp_id)
        atomic_write_json(paths.metrics_json, metrics)
        atomic_write_text(paths.metrics_csv, metrics_to_frame(exp_id, metrics).to_csv(index=False))

    def read_metrics(self, exp_id: str) -> dict[str, Any] | None:
        return read_json(self.paths(exp_id).metrics_json)

    def write_metadata(self, exp_id: str, metadata: dict[str, Any]) -> None:
        atomic_write_json(self.paths(exp_id).metadata, metadata)

    def read_metadata(self, exp_id: str) -> dict[str, Any] | None:
        return read_json(self.paths(exp_id).metadata)

    def list_epoch_grids(self, exp_id: str) -> dict[int, Path]:
        samples = self.paths(exp_id).samples
        grids = {}
        for p in samples.glob("epoch_*.png"):
            m = re.match(r"epoch_(\d+)\.png$", p.name)
            if m:
                grids[int(m.group(1))] = p
        return dict(sorted(grids.items()))

    # -- summaries ---------------------------------------------------------
    def summary(self, exp_id: str) -> dict[str, Any]:
        """Flat row used for experiment tables (safe on partially written experiments)."""
        row: dict[str, Any] = {"experiment_id": exp_id}
        try:
            cfg = self.load_config(exp_id)
            row.update(model=cfg.model, epochs=cfg.epochs, batch_size=cfg.batch_size, latent_dim=cfg.latent_dim,
                       learning_rate=cfg.learning_rate, seed=cfg.seed, train_size=cfg.effective_train_size)
        except StorageError as exc:
            row["error"] = str(exc)
        status = self.read_status(exp_id)
        row.update(state=status.get("state"), created_at=status.get("created_at"), epoch=status.get("epoch"))
        metrics = self.read_metrics(exp_id)
        if metrics:
            row.update(flatten_headline_metrics(metrics))
        return row

    def summaries(self) -> pd.DataFrame:
        return pd.DataFrame([self.summary(i) for i in self.experiment_ids()])


# Headline metrics that are defined identically for both models (safe to compare).
COMPARABLE_METRICS = {
    "validity_rate": ("proxy.validity.valid_fraction", "higher"),
    "mean_confidence": ("proxy.validity.mean_max_confidence", "higher"),
    "class_entropy_norm": ("proxy.validity.class_entropy_normalized", "higher"),
    "diversity_feature_ratio": ("proxy.diversity.feature_ratio_to_real", "closer to 1"),
    "diversity_pixel_ratio": ("proxy.diversity.pixel_ratio_to_real", "closer to 1"),
    "fd_mnist": ("proxy.fd_mnist.value", "lower"),
    "fid": ("distribution.fid.value", "lower"),
    "train_seconds": ("runtime.train_seconds", "n/a"),
    "eval_seconds": ("runtime.eval_seconds", "n/a"),
}


def _get_path(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def flatten_headline_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {name: _get_path(metrics, path) for name, (path, _) in COMPARABLE_METRICS.items()}


def metrics_to_frame(exp_id: str, metrics: dict[str, Any]) -> pd.DataFrame:
    """Flatten nested metrics into rows of (experiment_id, category, metric, value)."""
    rows: list[dict[str, Any]] = []

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else key, value)
        elif isinstance(node, (list, tuple)):
            rows.append({"metric": prefix, "value": json.dumps(node)})
        else:
            rows.append({"metric": prefix, "value": node})

    walk("", metrics)
    df = pd.DataFrame(rows, columns=["metric", "value"])
    df.insert(0, "category", df["metric"].str.split(".").str[0])
    df.insert(0, "experiment_id", exp_id)
    return df
