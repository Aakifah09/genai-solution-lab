"""Evaluation pipeline: sample -> classify -> proxy metrics -> FD-MNIST -> FID.

All numbers are computed from the trained checkpoint at evaluation time. Metrics are
grouped by *kind* so the UI and docs never mix them up:

* ``objective``    - the model's own training objective (not comparable across models)
* ``proxy``        - classifier-based measurements, identical procedure for both models
* ``distribution`` - standard FID
* ``reference``    - the same proxies on real validation data, plus evaluator accuracy
* ``runtime``      - timings and hardware
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch

from src.data.preprocessing import uint8_to_unit, validate_unit_images
from src.evaluation.classifier import DigitClassifier, predict
from src.evaluation.fid import FIDUnavailableError, compute_fd_mnist, compute_fid
from src.evaluation.metrics import diversity_metrics, validity_metrics
from src.persistence.storage import utc_now
from src.training.trainer import generate_images, save_grid
from src.utils.config import SPLIT_SEED, ExperimentConfig, evaluator_dir

EVAL_SEED_OFFSET = 10_000  # evaluation samples use seed + offset, never the training noise


def objective_summary(model_name: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}
    last = history[-1]
    if model_name == "gan":
        return {
            "final_d_loss": last["d_loss"],
            "final_g_loss": last["g_loss"],
            "final_d_real_prob": last["d_real_prob"],
            "final_d_fake_prob": last["d_fake_prob"],
            "note": "Adversarial losses measure the G-vs-D game, not sample quality; they are not comparable to VAE losses.",
        }
    return {
        "final_train_neg_elbo": last["train_loss"],
        "final_train_recon": last["train_recon"],
        "final_train_kl": last["train_kl"],
        "final_val_neg_elbo": last["val_loss"],
        "final_val_recon": last["val_recon"],
        "final_val_kl": last["val_kl"],
        "val_recon_per_pixel": last["val_recon"] / 784,
        "reduction": "BCE summed over 784 pixels and KL summed over latent dims, then averaged over images (nats/image)",
    }


def evaluate_model(
    model: torch.nn.Module,
    config: ExperimentConfig,
    history: list[dict[str, Any]],
    classifier: DigitClassifier,
    classifier_report: dict[str, Any],
    real_val_images_u8: torch.Tensor,
    device: torch.device,
    samples_dir=None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    start = time.perf_counter()
    n = config.eval_num_samples
    if len(real_val_images_u8) < n:
        raise ValueError(f"need {n} real validation images, only {len(real_val_images_u8)} available")
    sample_seed = config.seed + EVAL_SEED_OFFSET

    log(f"[eval] generating {n} samples (seed {sample_seed})")
    gen01 = generate_images(model, n, sample_seed, device)
    validate_unit_images(gen01, "generated images")
    real01 = uint8_to_unit(real_val_images_u8[:n])

    gen_probs, gen_feats = predict(classifier, gen01, device)
    real_probs, real_feats = predict(classifier, real01, device)

    metrics: dict[str, Any] = {
        "model": config.model,
        "objective": objective_summary(config.model, history),
        "proxy": {
            "validity": validity_metrics(gen_probs, config.validity_threshold),
            "diversity": diversity_metrics(gen01, gen_feats, real01, real_feats),
            "fd_mnist": compute_fd_mnist(gen_feats, real_feats),
        },
        "reference": {
            "real_validity": validity_metrics(real_probs, config.validity_threshold),
            "classifier_val_accuracy": classifier_report.get("best_val_accuracy"),
            "classifier_test_accuracy": classifier_report.get("test_accuracy"),
        },
    }
    log(f"[eval] validity={metrics['proxy']['validity']['valid_fraction']:.4f} "
        f"FD-MNIST={metrics['proxy']['fd_mnist']['value']:.3f}")

    if config.compute_fid:
        fid_start = time.perf_counter()
        try:
            log("[eval] computing standard FID with Inception-v3 features (slow on CPU)")
            m = config.fid_num_samples
            cache = evaluator_dir() / f"fid_ref_val{config.val_size}_split{SPLIT_SEED}_n{m}.npz"
            fid = compute_fid(gen01[:m], real01[:m], device, cache_path=cache)
            fid["seconds"] = time.perf_counter() - fid_start
            log(f"[eval] FID={fid['value']:.3f} ({fid['seconds']:.1f}s)")
        except FIDUnavailableError as exc:
            log(f"[eval] FID unavailable: {exc}")
            fid = {"status": "unavailable", "value": None, "reason": str(exc)}
    else:
        fid = {"status": "skipped", "value": None, "reason": "compute_fid=false in experiment config"}
    metrics["distribution"] = {"fid": fid}

    if samples_dir is not None:
        save_grid(gen01[:64], samples_dir / "eval_samples.png")
        save_grid(real01[:64], samples_dir / "real_reference.png")

    metrics["evaluation"] = {
        "num_samples": n,
        "sample_seed": sample_seed,
        "reference_split": "validation (never used for generative training)",
        "validity_threshold": config.validity_threshold,
        "evaluated_at": utc_now(),
        "eval_seconds": time.perf_counter() - start,
    }
    return metrics
