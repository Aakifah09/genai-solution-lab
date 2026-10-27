"""Experiment configuration: project paths plus a typed, validated config object.

All filesystem locations are resolved through functions (not module constants) so
tests and alternative deployments can redirect them with environment variables:

* ``GENAI_LAB_DATA_DIR``   - where MNIST is downloaded (default ``<project>/data``)
* ``GENAI_LAB_OUTPUT_DIR`` - experiments, evaluator artefacts, test reports
  (default ``<project>/outputs``)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The train/validation split of the 60k MNIST training images uses this fixed seed
# for *every* experiment, so all models share the exact same held-out validation set
# and metrics stay comparable. It is deliberately independent of the experiment seed.
SPLIT_SEED = 2024
MNIST_TRAIN_POOL = 60_000
SUPPORTED_MODELS = ("gan", "vae")


def data_dir() -> Path:
    return Path(os.environ.get("GENAI_LAB_DATA_DIR", PROJECT_ROOT / "data"))


def outputs_dir() -> Path:
    return Path(os.environ.get("GENAI_LAB_OUTPUT_DIR", PROJECT_ROOT / "outputs"))


def experiments_dir() -> Path:
    return outputs_dir() / "experiments"


def evaluator_dir() -> Path:
    return outputs_dir() / "evaluator"


def test_reports_dir() -> Path:
    return outputs_dir() / "test_reports"


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid. Lists every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Invalid experiment configuration:\n  - " + "\n  - ".join(problems))


# Per-model defaults. GAN settings follow the DCGAN paper (Adam lr=2e-4, beta1=0.5);
# VAE settings are the common Adam defaults for small convolutional VAEs.
_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "gan": {"latent_dim": 64, "learning_rate": 2e-4, "beta1": 0.5},
    "vae": {"latent_dim": 16, "learning_rate": 1e-3, "beta1": 0.9},
}


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce one training + evaluation run."""

    model: str
    epochs: int = 5
    batch_size: int = 128
    latent_dim: int = 64
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    seed: int = 42
    train_limit: int | None = None  # None = full training split (55,000 images)
    val_size: int = 5_000
    # GAN only: one-sided label smoothing, real targets become 1 - label_smoothing.
    label_smoothing: float = 0.1
    # VAE only: weight on the KL term (beta-VAE); 1.0 is the standard ELBO.
    kl_weight: float = 1.0
    # Evaluation
    eval_num_samples: int = 2_000
    validity_threshold: float = 0.9
    compute_fid: bool = True
    # Inception-v3 is slow on CPU (measured 1.0-7 images/s on the dev laptop CPU), so FID uses its own,
    # smaller sample count: the first fid_num_samples of the evaluation samples.
    fid_num_samples: int = 1_000
    deterministic: bool = True

    @classmethod
    def for_model(cls, model: str, **overrides: Any) -> "ExperimentConfig":
        """Build a config with model-appropriate defaults, then apply overrides."""
        if model not in SUPPORTED_MODELS:
            raise ConfigError([f"model must be one of {SUPPORTED_MODELS}, got {model!r}"])
        values: dict[str, Any] = {"model": model, **_MODEL_DEFAULTS[model]}
        values.update({k: v for k, v in overrides.items() if v is not None or k == "train_limit"})
        return cls.from_dict(values)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError([f"unknown configuration keys: {unknown}"])
        if "model" not in data:
            raise ConfigError(["'model' is required"])
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def train_split_size(self) -> int:
        return MNIST_TRAIN_POOL - self.val_size

    @property
    def effective_train_size(self) -> int:
        return self.train_split_size if self.train_limit is None else self.train_limit

    def validate(self) -> None:
        problems: list[str] = []

        def check_int(name: str, lo: int, hi: int) -> None:
            value = getattr(self, name)
            # bool is a subclass of int; reject it explicitly.
            if not isinstance(value, int) or isinstance(value, bool):
                problems.append(f"{name} must be an integer, got {value!r}")
            elif not lo <= value <= hi:
                problems.append(f"{name} must be between {lo} and {hi}, got {value}")

        def check_float(name: str, lo: float, hi: float, lo_inclusive: bool = True, hi_inclusive: bool = True) -> None:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"{name} must be a number, got {value!r}")
                return
            ok_lo = value >= lo if lo_inclusive else value > lo
            ok_hi = value <= hi if hi_inclusive else value < hi
            if not (ok_lo and ok_hi):
                lb, rb = ("[" if lo_inclusive else "("), ("]" if hi_inclusive else ")")
                problems.append(f"{name} must be in {lb}{lo}, {hi}{rb}, got {value}")

        if self.model not in SUPPORTED_MODELS:
            problems.append(f"model must be one of {SUPPORTED_MODELS}, got {self.model!r}")
        check_int("epochs", 1, 500)
        # BatchNorm in the GAN needs at least 2 samples per batch.
        check_int("batch_size", 2, 4096)
        check_int("latent_dim", 2, 1024)
        check_float("learning_rate", 0.0, 1.0, lo_inclusive=False)
        check_float("beta1", 0.0, 1.0, hi_inclusive=False)
        check_float("beta2", 0.0, 1.0, hi_inclusive=False)
        check_int("seed", 0, 2**32 - 1)
        check_int("val_size", 100, 20_000)
        check_float("label_smoothing", 0.0, 0.5, hi_inclusive=False)
        check_float("kl_weight", 0.0, 100.0)
        check_int("eval_num_samples", 100, 10_000)
        check_int("fid_num_samples", 100, 10_000)
        check_float("validity_threshold", 0.0, 1.0, lo_inclusive=False, hi_inclusive=False)
        for flag in ("compute_fid", "deterministic"):
            if not isinstance(getattr(self, flag), bool):
                problems.append(f"{flag} must be true or false")

        if self.train_limit is not None:
            check_int("train_limit", 1, MNIST_TRAIN_POOL)
        # Cross-field checks only make sense once the individual fields are valid.
        if not problems:
            if self.train_limit is not None:
                if self.train_limit > self.train_split_size:
                    problems.append(
                        f"train_limit ({self.train_limit}) exceeds the training split "
                        f"({self.train_split_size} = 60000 - val_size {self.val_size})"
                    )
                if self.train_limit < self.batch_size:
                    problems.append(
                        f"train_limit ({self.train_limit}) must be >= batch_size ({self.batch_size}) "
                        "so at least one full batch exists"
                    )
            if self.eval_num_samples > self.val_size:
                problems.append(
                    f"eval_num_samples ({self.eval_num_samples}) cannot exceed val_size ({self.val_size}): "
                    "generated samples are compared against the same number of real validation images"
                )
            if self.fid_num_samples > self.eval_num_samples:
                problems.append(
                    f"fid_num_samples ({self.fid_num_samples}) cannot exceed eval_num_samples ({self.eval_num_samples})"
                )
        if problems:
            raise ConfigError(problems)
