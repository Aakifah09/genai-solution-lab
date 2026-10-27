"""Convolutional Variational Autoencoder for 28x28 grayscale images.

Objective (per image, minimised): the negative ELBO

    L(x) = E_{q(z|x)}[-log p(x|z)]  +  beta * KL(q(z|x) || N(0, I))
           \\_____ reconstruction ____/        \\______ regularisation ______/

* p(x|z) is a product of independent Bernoullis over the 784 pixels, so the
  reconstruction term is binary cross-entropy **summed over pixels**.
* q(z|x) = N(mu, diag(sigma^2)) gives the closed-form KL
  -0.5 * sum_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2), **summed over latent dims**.
* Both terms are then **averaged over the batch**, so reported values are
  "nats per image". beta = 1 is the standard ELBO; beta > 1 trades reconstruction
  sharpness for a smoother, more Gaussian latent space.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

LOGVAR_CLAMP = (-10.0, 10.0)  # keeps exp(logvar) finite early in training


class VAEOutput(NamedTuple):
    recon: torch.Tensor  # pixel probabilities in [0, 1]
    logits: torch.Tensor  # pre-sigmoid decoder output (used by the loss)
    mu: torch.Tensor
    logvar: torch.Tensor


class VAELoss(NamedTuple):
    total: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden: int = 256):
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 64 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 28x28 logits
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4 or tuple(x.shape[1:]) != (1, 28, 28):
            raise ValueError(f"expected images shaped (N, 1, 28, 28), got {tuple(x.shape)}")
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(*LOGVAR_CLAMP)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        """z = mu + sigma * eps with eps ~ N(0, I): sampling becomes differentiable w.r.t. mu and logvar."""
        if eps is None:
            eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"expected latent shape (N, {self.latent_dim}), got {tuple(z.shape)}")
        return self.decoder(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x: torch.Tensor, eps: torch.Tensor | None = None) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, eps)
        logits = self.decode_logits(z)
        return VAEOutput(torch.sigmoid(logits), logits, mu, logvar)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu", generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw z ~ N(0, I) from the prior and decode to [0, 1] images."""
        z = torch.randn(n, self.latent_dim, device=device, generator=generator)
        return self.decode(z)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)) summed over latent dims, averaged over the batch."""
    per_image = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return per_image.mean()


def reconstruction_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Bernoulli negative log-likelihood (BCE) summed over pixels, averaged over the batch."""
    if target.min() < 0 or target.max() > 1:
        raise ValueError("VAE targets must be in [0, 1]")
    per_image = F.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten(1).sum(dim=1)
    return per_image.mean()


def vae_loss(out: VAEOutput, target: torch.Tensor, kl_weight: float = 1.0) -> VAELoss:
    recon = reconstruction_loss(out.logits, target)
    kl = kl_divergence(out.mu, out.logvar)
    return VAELoss(recon + kl_weight * kl, recon, kl)
