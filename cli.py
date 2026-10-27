"""Command-line entry point for GenAI Solution Lab.

Examples (PowerShell)::

    python cli.py env                                   # Python / torch / CUDA report
    python cli.py download-data                         # fetch MNIST into ./data
    python cli.py train-classifier                      # held-out evaluator (cached)
    python cli.py train --model gan --epochs 10
    python cli.py train --model vae --epochs 10 --train-limit 10000 --no-fid
    python cli.py resume --experiment-id <id>
    python cli.py evaluate --experiment-id <id>         # recompute metrics from checkpoint
    python cli.py list
    python cli.py export --output results.csv
    python cli.py test                                  # pytest + JUnit report for the dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from src.persistence.storage import ExperimentStore, utc_now


def cmd_env(_: argparse.Namespace) -> int:
    from src.utils.reproducibility import environment_info, get_device

    info = environment_info(get_device())
    print(json.dumps(info, indent=2))
    if not info["cuda_available"]:
        print("\nCUDA not available: training will run on CPU. See README > CUDA verification.")
    return 0


def cmd_download(_: argparse.Namespace) -> int:
    from src.data.dataset import load_mnist_raw
    from src.utils.config import data_dir

    train, _ = load_mnist_raw(train=True)
    test, _ = load_mnist_raw(train=False)
    print(f"MNIST ready in {data_dir()}: {len(train)} training images, {len(test)} test images")
    return 0


def cmd_train_classifier(args: argparse.Namespace) -> int:
    from src.evaluation.classifier import get_or_train_classifier
    from src.utils.reproducibility import get_device

    _, report = get_or_train_classifier(get_device(args.device), force=args.force)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0


def _run_foreground(store: ExperimentStore, exp_id: str, resume: bool, device: str) -> int:
    from src.training.experiment import run_experiment
    from src.training.trainer import TrainingCancelled

    # Register this process like a dashboard job so the UI sees it and blocks duplicate launches.
    store.write_job(exp_id, {"pid": os.getpid(), "command": sys.argv, "launched_at": utc_now(), "resume": resume, "source": "cli"})
    print(f"Experiment {exp_id} -> {store.paths(exp_id).root}")
    try:
        metrics = run_experiment(exp_id, store, resume=resume, device_preference=device)
    except TrainingCancelled:
        print("Cancelled; resume with: python cli.py resume --experiment-id", exp_id)
        return 2
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}\nFull log: {store.paths(exp_id).log}", file=sys.stderr)
        return 1
    headline = {
        "validity_rate": metrics["proxy"]["validity"]["valid_fraction"],
        "fd_mnist": metrics["proxy"]["fd_mnist"]["value"],
        "fid": metrics["distribution"]["fid"].get("value"),
        "train_seconds": metrics["runtime"]["train_seconds"],
        "eval_seconds": metrics["runtime"]["eval_seconds"],
        "device": metrics["runtime"]["device"],
    }
    print(json.dumps(headline, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from src.training.jobs import active_job
    from src.utils.config import ConfigError, ExperimentConfig

    store = ExperimentStore()
    running = active_job(store)
    if running:
        print(f"Experiment {running} is already running; wait for it or cancel it from the dashboard.", file=sys.stderr)
        return 1
    try:
        config = ExperimentConfig.for_model(
            args.model, epochs=args.epochs, batch_size=args.batch_size, latent_dim=args.latent_dim,
            learning_rate=args.lr, seed=args.seed, train_limit=args.train_limit, kl_weight=args.kl_weight,
            label_smoothing=args.label_smoothing, eval_num_samples=args.eval_samples,
            compute_fid=not args.no_fid, fid_num_samples=args.fid_samples,
        )
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    exp_id = store.create(config)
    return _run_foreground(store, exp_id, resume=False, device=args.device)


def cmd_resume(args: argparse.Namespace) -> int:
    store = ExperimentStore()
    return _run_foreground(store, args.experiment_id, resume=True, device=args.device)


def cmd_evaluate(args: argparse.Namespace) -> int:
    from src.training.experiment import reevaluate_experiment

    metrics = reevaluate_experiment(args.experiment_id, device_preference=args.device)
    print(json.dumps({k: metrics[k] for k in ("proxy", "distribution", "runtime")}, indent=2, default=str))
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    import pandas as pd

    df = ExperimentStore().summaries()
    if df.empty:
        print("No experiments yet. Run: python cli.py train --model gan")
        return 0
    cols = [c for c in ["experiment_id", "model", "state", "epochs", "train_size", "validity_rate", "fd_mnist", "fid", "train_seconds"] if c in df]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[cols].to_string(index=False))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import pandas as pd

    store = ExperimentStore()
    frames = [pd.read_csv(store.paths(i).metrics_csv) for i in store.experiment_ids() if store.paths(i).metrics_csv.exists()]
    if not frames:
        print("No evaluated experiments to export.", file=sys.stderr)
        return 1
    pd.concat(frames).to_csv(args.output, index=False)
    print(f"Wrote {sum(len(f) for f in frames)} metric rows from {len(frames)} experiments to {args.output}")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    from src.utils.config import test_reports_dir

    report = test_reports_dir() / "junit.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}", *args.pytest_args]
    print(" ".join(cmd))
    code = subprocess.call(cmd)
    print(f"JUnit report: {report}")
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="GenAI Solution Lab: GAN/VAE training and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="report Python, PyTorch and CUDA information").set_defaults(func=cmd_env)
    sub.add_parser("download-data", help="download MNIST").set_defaults(func=cmd_download)

    p = sub.add_parser("train-classifier", help="train/cache the held-out evaluator classifier")
    p.add_argument("--force", action="store_true", help="retrain even if a cached classifier exists")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.set_defaults(func=cmd_train_classifier)

    p = sub.add_parser("train", help="train and evaluate a GAN or VAE in the foreground")
    p.add_argument("--model", required=True, choices=["gan", "vae"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--latent-dim", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--train-limit", type=int, help="use only the first N training-split images")
    p.add_argument("--kl-weight", type=float, help="VAE only (beta)")
    p.add_argument("--label-smoothing", type=float, help="GAN only")
    p.add_argument("--eval-samples", type=int)
    p.add_argument("--fid-samples", type=int, help="images per side for FID (default 1000; slow on CPU)")
    p.add_argument("--no-fid", action="store_true", help="skip Inception FID (FD-MNIST is always computed)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("resume", help="resume a cancelled/interrupted experiment from its checkpoint")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("evaluate", help="recompute metrics for a finished experiment")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.set_defaults(func=cmd_evaluate)

    sub.add_parser("list", help="list experiments").set_defaults(func=cmd_list)

    p = sub.add_parser("export", help="export all metrics to one CSV")
    p.add_argument("--output", default="results.csv")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("test", help="run pytest and save a JUnit report shown in the dashboard")
    p.add_argument("pytest_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
