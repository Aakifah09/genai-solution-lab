"""Seeding, device selection and environment capture."""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices).

    With ``deterministic=True`` cuDNN autotuning is disabled and PyTorch is asked to
    prefer deterministic kernels. Ops without a deterministic implementation only
    warn (``warn_only=True``) instead of crashing a long run, so bit-exact results
    are guaranteed on CPU but only best-effort on GPU.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seeded_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """An isolated RNG stream, so sampling for evaluation never perturbs training RNG."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def get_device(preference: str = "auto") -> torch.device:
    """Return CUDA when available (or requested), otherwise CPU.

    ``preference`` may be ``"auto"``, ``"cpu"`` or ``"cuda"``. Requesting CUDA on a
    machine without it raises instead of silently falling back.
    """
    preference = preference.lower()
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                f"Installed torch build: {torch.__version__}. Install a CUDA-enabled PyTorch "
                "wheel (see README > CUDA verification) or use --device cpu."
            )
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError(f"device must be 'auto', 'cpu' or 'cuda', got {preference!r}")


def device_description(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return f"cpu ({platform.processor() or platform.machine()}, {torch.get_num_threads()} torch threads)"


def environment_info(device: torch.device | None = None) -> dict[str, Any]:
    """Software and hardware facts recorded with every experiment."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        # Read the version from package metadata: importing torchvision just for its version
        # string costs seconds (over a minute on a cold, freshly booted Windows machine).
        torchvision_version = version("torchvision")
    except PackageNotFoundError:
        torchvision_version = None
    device = device or get_device()
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": device.type,
        "device_description": device_description(device),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
    return info
