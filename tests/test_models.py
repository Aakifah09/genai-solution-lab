import math

import pytest
import torch

from src.models.gan import Discriminator, Generator, build_gan, sample_latent
from src.models.vae import VAE, kl_divergence, reconstruction_loss, vae_loss


# -- GAN -----------------------------------------------------------------------

@pytest.mark.parametrize("latent_dim", [2, 64, 100])
def test_generator_output_shape_and_tanh_range(latent_dim):
    gen, _ = build_gan(latent_dim)
    out = gen(sample_latent(8, latent_dim))
    assert out.shape == (8, 1, 28, 28)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_generator_rejects_wrong_latent_shape():
    gen = Generator(latent_dim=16)
    with pytest.raises(ValueError, match="latent shape"):
        gen(torch.randn(4, 32))


def test_discriminator_returns_one_logit_per_image():
    disc = Discriminator()
    logits = disc(torch.rand(5, 1, 28, 28) * 2 - 1)
    assert logits.shape == (5,)
    assert torch.isfinite(logits).all()


def test_discriminator_rejects_wrong_image_shape():
    with pytest.raises(ValueError, match="1, 28, 28"):
        Discriminator()(torch.rand(2, 3, 28, 28))


def test_gan_gradients_reach_generator_through_discriminator():
    gen, disc = build_gan(8)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(disc(gen(torch.randn(4, 8))), torch.ones(4))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gen.parameters())


# -- VAE -----------------------------------------------------------------------

def test_vae_reconstruction_shape_and_range():
    vae = VAE(latent_dim=4)
    x = torch.rand(6, 1, 28, 28)
    out = vae(x)
    assert out.recon.shape == x.shape
    assert out.recon.min() >= 0 and out.recon.max() <= 1
    assert out.mu.shape == (6, 4) and out.logvar.shape == (6, 4)
    assert vae.sample(3).shape == (3, 1, 28, 28)


def test_kl_is_zero_for_standard_normal_posterior():
    mu, logvar = torch.zeros(3, 5), torch.zeros(3, 5)
    assert kl_divergence(mu, logvar).item() == pytest.approx(0.0)


def test_kl_matches_closed_form():
    # One latent dim, mu=1, sigma^2=e  ->  KL = 0.5 * (sigma^2 + mu^2 - 1 - log sigma^2) = 0.5 * (e + 1 - 1 - 1)
    mu, logvar = torch.tensor([[1.0]]), torch.tensor([[1.0]])
    assert kl_divergence(mu, logvar).item() == pytest.approx(0.5 * (math.e - 1))


def test_reconstruction_loss_sums_pixels_and_averages_batch():
    logits = torch.zeros(2, 1, 28, 28)  # sigmoid(0) = 0.5 -> BCE = ln 2 per pixel
    target = torch.ones(2, 1, 28, 28)
    assert reconstruction_loss(logits, target).item() == pytest.approx(784 * math.log(2), rel=1e-5)


def test_reconstruction_loss_rejects_out_of_range_targets():
    with pytest.raises(ValueError):
        reconstruction_loss(torch.zeros(1, 1, 28, 28), torch.full((1, 1, 28, 28), -1.0))


def test_vae_loss_combines_terms_with_kl_weight_and_has_gradients():
    torch.manual_seed(0)
    vae = VAE(latent_dim=3)
    x = torch.rand(4, 1, 28, 28)
    out = vae(x)
    loss = vae_loss(out, x, kl_weight=2.5)
    assert loss.total.item() == pytest.approx(loss.recon.item() + 2.5 * loss.kl.item(), rel=1e-5)
    loss.total.backward()
    for name, p in vae.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert vae.fc_mu.weight.grad.abs().sum() > 0
    assert vae.fc_logvar.weight.grad.abs().sum() > 0


def test_reparameterize_uses_given_noise():
    mu, logvar = torch.tensor([[1.0, -1.0]]), torch.log(torch.tensor([[4.0, 1.0]]))
    z = VAE.reparameterize(mu, logvar, eps=torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(z, torch.tensor([[3.0, 1.0]]))
