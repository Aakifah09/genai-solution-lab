"""DCGAN-style Generator and Discriminator for 28x28 grayscale images.

Generator:      z (latent_dim) -> Linear -> 128x7x7 -> ConvT -> 64x14x14 -> ConvT -> 1x28x28, Tanh
Discriminator:  1x28x28 -> Conv -> 64x14x14 -> Conv -> 128x7x7 -> Linear -> 1 logit

The discriminator returns raw logits; training uses ``BCEWithLogitsLoss`` which is
numerically more stable than Sigmoid + BCELoss.
"""

from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 64):
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = latent_dim
        c = base_channels
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 2 * c * 7 * 7, bias=False),
            nn.BatchNorm1d(2 * c * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.Unflatten(1, (2 * c, 7, 7)),
            nn.ConvTranspose2d(2 * c, c, kernel_size=4, stride=2, padding=1, bias=False),  # 14x14
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, 1, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"expected latent shape (N, {self.latent_dim}), got {tuple(z.shape)}")
        return self.upsample(self.project(z))


class Discriminator(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 2 * c, kernel_size=4, stride=2, padding=1, bias=False),  # 7x7
            nn.BatchNorm2d(2 * c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(2 * c * 7 * 7, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one real-vs-fake logit per image, shape (N,)."""
        if x.dim() != 4 or tuple(x.shape[1:]) != (1, 28, 28):
            raise ValueError(f"expected images shaped (N, 1, 28, 28), got {tuple(x.shape)}")
        return self.net(x).squeeze(1)


def init_dcgan_weights(module: nn.Module) -> None:
    """DCGAN initialisation: conv/linear ~ N(0, 0.02), batch-norm gamma ~ N(1, 0.02)."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def sample_latent(n: int, latent_dim: int, device: torch.device | str = "cpu", generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(n, latent_dim, device=device, generator=generator)


def build_gan(latent_dim: int) -> tuple[Generator, Discriminator]:
    gen, disc = Generator(latent_dim), Discriminator()
    gen.apply(init_dcgan_weights)
    disc.apply(init_dcgan_weights)
    return gen, disc
