"""Experiment orchestration: config -> data -> training -> evaluation -> persistence.

Run as a module for background jobs launched by the dashboard::

    python -m src.training.experiment --experiment-id gan-20260101-120000-abc123 [--resume]

The process owns its experiment's status/history/metrics files for its lifetime.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from typing import Any

import torch

from src.data.dataset import load_splits, make_loader
from src.data.preprocessing import normalization_for_model, preprocessing_summary
from src.evaluation.classifier import get_or_train_classifier
from src.evaluation.evaluate import evaluate_model
from src.evaluation.plots import plot_training_curves
from src.persistence.storage import ExperimentStore, utc_now
from src.training.trainer import (
    TrainingCancelled,
    TrainingContext,
    TrainingDivergedError,
    load_checkpoint,
    load_generative_model,
    train_gan,
    train_vae,
)
from src.utils.config import SPLIT_SEED
from src.utils.logging_utils import add_file_handler, get_logger
from src.utils.reproducibility import environment_info, get_device, set_seed

STATUS_THROTTLE_SECONDS = 1.0


def _metadata(store: ExperimentStore, exp_id: str, device: torch.device, runtime: dict[str, Any]) -> dict[str, Any]:
    cfg = store.load_config(exp_id)
    return {
        "experiment_id": exp_id,
        "model_name": {"gan": "DCGAN-style GAN", "vae": "Convolutional VAE"}[cfg.model],
        "config": cfg.to_dict(),
        "dataset": {
            "name": "MNIST",
            "train_size": cfg.effective_train_size,
            "val_size": cfg.val_size,
            "split_seed": SPLIT_SEED,
            "test_set_used_for_training": False,
            "preprocessing": preprocessing_summary(cfg.model),
        },
        "optimizer": {"name": "Adam", "lr": cfg.learning_rate, "betas": [cfg.beta1, cfg.beta2]},
        "environment": environment_info(device),
        "runtime": runtime,
        "written_at": utc_now(),
    }


def run_experiment(exp_id: str, store: ExperimentStore | None = None, resume: bool = False, device_preference: str = "auto") -> dict[str, Any]:
    """Train and evaluate one experiment. Returns the metrics dict. Status is always left terminal."""
    store = store or ExperimentStore()
    paths = store.paths(exp_id)
    logger = get_logger(f"genai_lab.{exp_id}")
    handler = add_file_handler(logger, paths.log)
    log = logger.info
    last_write = [0.0]

    def on_progress(progress: dict[str, Any]) -> None:
        now = time.monotonic()
        if now - last_write[0] >= STATUS_THROTTLE_SECONDS or progress["batch"] == progress["batches"]:
            last_write[0] = now
            store.update_status(exp_id, state="running", **progress)

    try:
        config = store.load_config(exp_id)
        device = get_device(device_preference)
        set_seed(config.seed, config.deterministic)
        store.clear_cancel(exp_id)
        store.update_status(exp_id, state="running", pid=os.getpid(), started_at=utc_now(), device=device.type,
                            error=None, epoch=0, epochs=config.epochs)
        log(f"Experiment {exp_id} | model={config.model} | device={device} | resume={resume}")
        log(f"Config: {config.to_dict()}")

        resume_state = None
        if resume:
            if not paths.checkpoint.exists():
                raise FileNotFoundError("cannot resume: no checkpoint was saved before the run stopped")
            resume_state = load_checkpoint(paths.checkpoint)

        splits = load_splits(val_size=config.val_size, train_limit=config.train_limit)
        log(f"Data: train={splits.sizes['train']} val={splits.sizes['val']} (test split not loaded)")
        norm = normalization_for_model(config.model)
        train_loader = make_loader(splits.train_images, splits.train_labels, norm, config.batch_size,
                                   shuffle=True, seed=config.seed, drop_last=config.model == "gan")
        ctx = TrainingContext(config=config, device=device, paths=paths, on_progress=on_progress,
                              on_epoch_end=lambda h: store.write_history(exp_id, h),
                              should_cancel=lambda: store.cancel_requested(exp_id), log=log)

        if resume_state and resume_state["epoch"] >= config.epochs:
            log("Checkpoint already contains all epochs; skipping straight to evaluation")
            history, train_seconds = resume_state["history"], resume_state["train_seconds"]
            model = load_generative_model(resume_state, device)
        elif config.model == "gan":
            result = train_gan(ctx, train_loader, resume_state)
            history, train_seconds, model = result.history, result.train_seconds, result.models["generator"]
        else:
            val_loader = make_loader(splits.val_images, splits.val_labels, norm, 500, shuffle=False, seed=config.seed)
            result = train_vae(ctx, train_loader, val_loader, resume_state)
            history, train_seconds, model = result.history, result.train_seconds, result.models["vae"]
        store.write_history(exp_id, history)
        plot_training_curves(history, config.model, paths.samples / "training_curves.png")

        store.update_status(exp_id, state="evaluating", epoch=config.epochs)
        classifier, clf_report = get_or_train_classifier(device, val_size=config.val_size, log=log)
        metrics = evaluate_model(model, config, history, classifier, clf_report, splits.val_images, device,
                                 samples_dir=paths.samples, log=log)
        runtime = {"train_seconds": train_seconds, "eval_seconds": metrics["evaluation"]["eval_seconds"],
                   "device": device.type, "device_description": environment_info(device)["device_description"]}
        metrics["runtime"] = runtime
        store.write_metrics(exp_id, metrics)
        store.write_metadata(exp_id, _metadata(store, exp_id, device, runtime))
        store.update_status(exp_id, state="completed", finished_at=utc_now())
        log(f"Completed in {train_seconds:.1f}s training + {runtime['eval_seconds']:.1f}s evaluation")
        return metrics

    except TrainingCancelled as exc:
        log(f"Cancelled: {exc}. Last completed epoch checkpoint kept for resuming.")
        store.update_status(exp_id, state="cancelled", finished_at=utc_now(), error=str(exc),
                            resumable=paths.checkpoint.exists())
        store.clear_cancel(exp_id)
        raise
    except Exception as exc:
        logger.error("Experiment failed:\n%s", traceback.format_exc())
        hint = ""
        if isinstance(exc, TrainingDivergedError):
            hint = "Training diverged; change the hyperparameters and start a new experiment."
        store.update_status(exp_id, state="failed", finished_at=utc_now(), error=f"{type(exc).__name__}: {exc}",
                            hint=hint, resumable=paths.checkpoint.exists() and not isinstance(exc, TrainingDivergedError))
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


def reevaluate_experiment(exp_id: str, store: ExperimentStore | None = None, device_preference: str = "auto") -> dict[str, Any]:
    """Recompute metrics from the saved checkpoint (e.g. after enabling FID)."""
    store = store or ExperimentStore()
    paths = store.paths(exp_id)
    config = store.load_config(exp_id)
    device = get_device(device_preference)
    checkpoint = load_checkpoint(paths.checkpoint)
    if checkpoint["epoch"] < config.epochs:
        raise RuntimeError(f"checkpoint has {checkpoint['epoch']}/{config.epochs} epochs; resume training first")
    model = load_generative_model(checkpoint, device)
    splits = load_splits(val_size=config.val_size, train_limit=config.train_limit)
    classifier, clf_report = get_or_train_classifier(device, val_size=config.val_size)
    metrics = evaluate_model(model, config, checkpoint["history"], classifier, clf_report, splits.val_images, device,
                             samples_dir=paths.samples)
    plot_training_curves(checkpoint["history"], config.model, paths.samples / "training_curves.png")
    runtime = {"train_seconds": checkpoint["train_seconds"], "eval_seconds": metrics["evaluation"]["eval_seconds"],
               "device": device.type, "device_description": environment_info(device)["device_description"]}
    metrics["runtime"] = runtime
    store.write_metrics(exp_id, metrics)
    store.write_metadata(exp_id, _metadata(store, exp_id, device, runtime))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one stored experiment (used by the dashboard job launcher).")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        run_experiment(args.experiment_id, resume=args.resume, device_preference=args.device)
    except TrainingCancelled:
        return 2
    except Exception:  # already logged and written to status.json by run_experiment
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
