"""Training loops for the GAN and the VAE.

Both loops share the same contract:

* ``ctx.on_progress(dict)`` is called every few batches with live losses/timing.
* ``ctx.should_cancel()`` is polled every batch; cancellation discards the partial
  epoch and raises ``TrainingCancelled`` (the last completed epoch's checkpoint stays
  on disk, so the run can be resumed).
* After every epoch: history row, sample grid(s) and a resumable checkpoint
  (models + optimizers + history + RNG state) are written.
* Non-finite losses raise ``TrainingDivergedError`` with concrete suggestions.

Batch order is reseeded per epoch (seed + epoch), so a resumed run sees exactly the
same batches it would have seen without interruption.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from src.data.preprocessing import to_unit_range
from src.models.gan import Generator, build_gan, sample_latent
from src.models.vae import VAE, vae_loss
from src.persistence.storage import ExperimentPaths, _replace_with_retry
from src.utils.config import ExperimentConfig
from src.utils.reproducibility import seeded_generator

GRID_SIZE = 64  # images per sample grid (8 x 8)
PROGRESS_EVERY = 10  # batches between progress callbacks


class TrainingCancelled(Exception):
    """Training stopped because cancellation was requested."""


class TrainingDivergedError(RuntimeError):
    """A loss became NaN/Inf."""


@dataclass
class TrainingContext:
    config: ExperimentConfig
    device: torch.device
    paths: ExperimentPaths
    on_progress: Callable[[dict[str, Any]], None] = lambda _: None
    on_epoch_end: Callable[[list[dict[str, Any]]], None] = lambda _: None
    should_cancel: Callable[[], bool] = lambda: False
    log: Callable[[str], None] = print


@dataclass
class TrainingResult:
    history: list[dict[str, Any]]
    train_seconds: float
    models: dict[str, torch.nn.Module] = field(default_factory=dict)


# -- file helpers ------------------------------------------------------------

def save_grid(images01: torch.Tensor, path: Path, nrow: int = 8) -> None:
    """Write a PNG grid atomically (dashboard may be reading the previous version)."""
    grid = make_grid(images01.detach().cpu().clamp(0, 1), nrow=nrow, padding=2, pad_value=0.25)
    array = (grid.mul(255).round().byte().permute(1, 2, 0).numpy())
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(buf.getvalue())
    _replace_with_retry(str(tmp), path)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    _replace_with_retry(str(tmp), path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    # weights_only=True refuses arbitrary pickled code; our payload is tensors + primitives.
    return torch.load(path, map_location="cpu", weights_only=True)


def _check_finite(losses: dict[str, float], epoch: int, batch: int, hint: str) -> None:
    bad = {k: v for k, v in losses.items() if not torch.isfinite(torch.tensor(v))}
    if bad:
        raise TrainingDivergedError(f"non-finite loss at epoch {epoch}, batch {batch}: {bad}. {hint}")


def _reseed_loader(loader: DataLoader, seed: int, epoch: int) -> None:
    if loader.generator is not None:
        loader.generator.manual_seed(seed + epoch)


# -- GAN -----------------------------------------------------------------------

def train_gan(ctx: TrainingContext, train_loader: DataLoader, resume: dict[str, Any] | None = None) -> TrainingResult:
    cfg, device = ctx.config, ctx.device
    gen, disc = build_gan(cfg.latent_dim)
    gen, disc = gen.to(device), disc.to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    # Fixed noise: the per-epoch grids show the *same* latent points evolving.
    fixed_z = sample_latent(GRID_SIZE, cfg.latent_dim, generator=seeded_generator(cfg.seed + 1)).to(device)

    history: list[dict[str, Any]] = []
    start_epoch, prior_seconds = 1, 0.0
    if resume:
        gen.load_state_dict(resume["state"]["generator"])
        disc.load_state_dict(resume["state"]["discriminator"])
        opt_g.load_state_dict(resume["state"]["opt_g"])
        opt_d.load_state_dict(resume["state"]["opt_d"])
        torch.set_rng_state(resume["rng_state"])
        history = list(resume["history"])
        start_epoch, prior_seconds = resume["epoch"] + 1, resume["train_seconds"]
        ctx.log(f"Resuming GAN from completed epoch {resume['epoch']}")

    real_target = 1.0 - cfg.label_smoothing
    hint = "GAN training diverged: try a lower learning_rate (e.g. 1e-4) or more label_smoothing."
    n_batches = len(train_loader)
    run_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        _reseed_loader(train_loader, cfg.seed, epoch)
        gen.train()
        disc.train()
        epoch_start = time.perf_counter()
        sums = {"d_loss": 0.0, "g_loss": 0.0, "d_real_prob": 0.0, "d_fake_prob": 0.0}
        seen = 0
        for batch_idx, (real, _) in enumerate(train_loader, start=1):
            if ctx.should_cancel():
                raise TrainingCancelled(f"cancelled during epoch {epoch}")
            real = real.to(device)
            b = real.size(0)
            fake = gen(sample_latent(b, cfg.latent_dim, device=device))

            # Discriminator: maximise log D(x) + log(1 - D(G(z))), with one-sided label smoothing.
            d_real = disc(real)
            d_fake = disc(fake.detach())
            d_loss = (F.binary_cross_entropy_with_logits(d_real, torch.full_like(d_real, real_target))
                      + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # Generator: non-saturating loss, maximise log D(G(z)).
            g_out = disc(fake)
            g_loss = F.binary_cross_entropy_with_logits(g_out, torch.ones_like(g_out))
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            losses = {"d_loss": d_loss.item(), "g_loss": g_loss.item()}
            _check_finite(losses, epoch, batch_idx, hint)
            sums["d_loss"] += losses["d_loss"] * b
            sums["g_loss"] += losses["g_loss"] * b
            sums["d_real_prob"] += torch.sigmoid(d_real.detach()).sum().item()
            sums["d_fake_prob"] += torch.sigmoid(d_fake.detach()).sum().item()
            seen += b
            if batch_idx % PROGRESS_EVERY == 0 or batch_idx == n_batches:
                ctx.on_progress({
                    "epoch": epoch, "epochs": cfg.epochs, "batch": batch_idx, "batches": n_batches,
                    "elapsed_seconds": prior_seconds + time.perf_counter() - run_start,
                    "losses": {k: v / seen for k, v in sums.items()},
                })

        epoch_seconds = time.perf_counter() - epoch_start
        row = {"epoch": epoch, **{k: v / seen for k, v in sums.items()}, "epoch_seconds": epoch_seconds,
               "elapsed_seconds": prior_seconds + time.perf_counter() - run_start}
        history.append(row)
        gen.eval()
        with torch.no_grad():
            save_grid(to_unit_range(gen(fixed_z), "tanh"), ctx.paths.epoch_grid(epoch))
        save_checkpoint(ctx.paths.checkpoint, {
            "model": "gan", "epoch": epoch, "history": history, "train_seconds": row["elapsed_seconds"],
            "config": cfg.to_dict(), "rng_state": torch.get_rng_state(),
            "state": {"generator": gen.state_dict(), "discriminator": disc.state_dict(),
                      "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()},
        })
        ctx.log(f"[gan] epoch {epoch}/{cfg.epochs} d_loss={row['d_loss']:.4f} g_loss={row['g_loss']:.4f} "
                f"D(x)={row['d_real_prob']:.3f} D(G(z))={row['d_fake_prob']:.3f} ({epoch_seconds:.1f}s)")
        ctx.on_epoch_end(history)

    total = history[-1]["elapsed_seconds"] if history else 0.0
    return TrainingResult(history, total, {"generator": gen.eval(), "discriminator": disc.eval()})


# -- VAE -----------------------------------------------------------------------

@torch.no_grad()
def evaluate_vae(model: VAE, loader: DataLoader, device: torch.device, kl_weight: float, seed: int) -> dict[str, float]:
    """Held-out loss with a fixed noise stream, so values are comparable across epochs."""
    model.eval()
    gen = seeded_generator(seed)
    sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    for x, _ in loader:
        x = x.to(device)
        eps = torch.randn(x.size(0), model.latent_dim, generator=gen).to(device)
        out = model(x, eps=eps)
        loss = vae_loss(out, x, kl_weight)
        b = x.size(0)
        sums["loss"] += float(loss.total) * b
        sums["recon"] += float(loss.recon) * b
        sums["kl"] += float(loss.kl) * b
        n += b
    return {k: v / n for k, v in sums.items()}


def train_vae(ctx: TrainingContext, train_loader: DataLoader, val_loader: DataLoader, resume: dict[str, Any] | None = None) -> TrainingResult:
    cfg, device = ctx.config, ctx.device
    model = VAE(cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    fixed_z = torch.randn(GRID_SIZE, cfg.latent_dim, generator=seeded_generator(cfg.seed + 1)).to(device)
    # 32 fixed validation images for the reconstruction grid (alternating rows: original / reconstruction).
    recon_batch = next(iter(val_loader))[0][:32].to(device)

    history: list[dict[str, Any]] = []
    start_epoch, prior_seconds = 1, 0.0
    if resume:
        model.load_state_dict(resume["state"]["vae"])
        opt.load_state_dict(resume["state"]["opt"])
        torch.set_rng_state(resume["rng_state"])
        history = list(resume["history"])
        start_epoch, prior_seconds = resume["epoch"] + 1, resume["train_seconds"]
        ctx.log(f"Resuming VAE from completed epoch {resume['epoch']}")

    hint = "VAE training diverged: try a lower learning_rate (e.g. 3e-4)."
    n_batches = len(train_loader)
    run_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        _reseed_loader(train_loader, cfg.seed, epoch)
        model.train()
        epoch_start = time.perf_counter()
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        seen = 0
        for batch_idx, (x, _) in enumerate(train_loader, start=1):
            if ctx.should_cancel():
                raise TrainingCancelled(f"cancelled during epoch {epoch}")
            x = x.to(device)
            out = model(x)
            loss = vae_loss(out, x, cfg.kl_weight)
            opt.zero_grad(set_to_none=True)
            loss.total.backward()
            opt.step()

            values = {"loss": loss.total.item(), "recon": loss.recon.item(), "kl": loss.kl.item()}
            _check_finite(values, epoch, batch_idx, hint)
            b = x.size(0)
            for k, v in values.items():
                sums[k] += v * b
            seen += b
            if batch_idx % PROGRESS_EVERY == 0 or batch_idx == n_batches:
                ctx.on_progress({
                    "epoch": epoch, "epochs": cfg.epochs, "batch": batch_idx, "batches": n_batches,
                    "elapsed_seconds": prior_seconds + time.perf_counter() - run_start,
                    "losses": {f"train_{k}": v / seen for k, v in sums.items()},
                })

        val = evaluate_vae(model, val_loader, device, cfg.kl_weight, cfg.seed + 2)
        epoch_seconds = time.perf_counter() - epoch_start
        row = {"epoch": epoch, **{f"train_{k}": v / seen for k, v in sums.items()},
               **{f"val_{k}": v for k, v in val.items()},
               "epoch_seconds": epoch_seconds, "elapsed_seconds": prior_seconds + time.perf_counter() - run_start}
        history.append(row)
        model.eval()
        with torch.no_grad():
            save_grid(model.decode(fixed_z), ctx.paths.epoch_grid(epoch))
            recon = model.decode(model.encode(recon_batch)[0])  # decode the posterior mean
            pairs = torch.stack([recon_batch.view(4, 8, 1, 28, 28), recon.view(4, 8, 1, 28, 28)], dim=1)
            save_grid(pairs.reshape(64, 1, 28, 28), ctx.paths.recon_grid(epoch))
        save_checkpoint(ctx.paths.checkpoint, {
            "model": "vae", "epoch": epoch, "history": history, "train_seconds": row["elapsed_seconds"],
            "config": cfg.to_dict(), "rng_state": torch.get_rng_state(),
            "state": {"vae": model.state_dict(), "opt": opt.state_dict()},
        })
        ctx.log(f"[vae] epoch {epoch}/{cfg.epochs} train_loss={row['train_loss']:.2f} "
                f"(recon={row['train_recon']:.2f}, kl={row['train_kl']:.2f}) val_loss={row['val_loss']:.2f} ({epoch_seconds:.1f}s)")
        ctx.on_epoch_end(history)

    total = history[-1]["elapsed_seconds"] if history else 0.0
    return TrainingResult(history, total, {"vae": model.eval()})


def load_generative_model(checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Rebuild the sampling model (GAN generator or VAE) from a checkpoint payload."""
    cfg = ExperimentConfig.from_dict(checkpoint["config"])
    if checkpoint["model"] == "gan":
        model: torch.nn.Module = Generator(cfg.latent_dim)
        model.load_state_dict(checkpoint["state"]["generator"])
    elif checkpoint["model"] == "vae":
        model = VAE(cfg.latent_dim)
        model.load_state_dict(checkpoint["state"]["vae"])
    else:
        raise ValueError(f"unknown model in checkpoint: {checkpoint['model']!r}")
    return model.to(device).eval()


@torch.no_grad()
def generate_images(model: torch.nn.Module, n: int, seed: int, device: torch.device, batch_size: int = 500) -> torch.Tensor:
    """Sample n images in [0, 1] from a GAN generator or VAE prior with a dedicated RNG stream."""
    rng = seeded_generator(seed)
    model.eval()
    chunks = []
    for start in range(0, n, batch_size):
        m = min(batch_size, n - start)
        if isinstance(model, Generator):
            z = torch.randn(m, model.latent_dim, generator=rng).to(device)
            chunks.append(to_unit_range(model(z), "tanh").cpu())
        elif isinstance(model, VAE):
            z = torch.randn(m, model.latent_dim, generator=rng).to(device)
            chunks.append(model.decode(z).cpu())
        else:
            raise TypeError(f"cannot sample from {type(model).__name__}")
    return torch.cat(chunks)
