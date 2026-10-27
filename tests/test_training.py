"""Training-loop, checkpoint, reproducibility and pipeline tests on tiny synthetic data."""


import pytest
import torch

from src.data.dataset import MNISTSplits, make_loader
from src.evaluation.classifier import DigitClassifier
from src.evaluation.evaluate import evaluate_model
from src.models.gan import Generator, build_gan
from src.models.vae import VAE
from src.training import experiment as experiment_module
from src.training.trainer import (
    TrainingCancelled,
    TrainingContext,
    TrainingDivergedError,
    generate_images,
    load_checkpoint,
    load_generative_model,
    save_checkpoint,
    train_gan,
    train_vae,
)
from src.utils.config import ExperimentConfig
from src.utils.reproducibility import get_device, seeded_generator, set_seed

CPU = torch.device("cpu")


def tiny_config(model, **kw):
    base = dict(epochs=2, batch_size=32, train_limit=128, val_size=100, eval_num_samples=100, fid_num_samples=100,
                compute_fid=False, latent_dim=8)
    base.update(kw)
    return ExperimentConfig.for_model(model, **base)


def loaders(synthetic_mnist, cfg):
    images, labels = synthetic_mnist
    norm = "tanh" if cfg.model == "gan" else "unit"
    train = make_loader(images[:128], labels[:128], norm, cfg.batch_size, shuffle=True, seed=cfg.seed, drop_last=cfg.model == "gan")
    val = make_loader(images[500:600], labels[500:600], norm, 50, shuffle=False, seed=cfg.seed)
    return train, val


def make_ctx(store, cfg, **kw):
    eid = store.create(cfg)
    return TrainingContext(config=cfg, device=CPU, paths=store.paths(eid), log=lambda _: None, **kw), eid


def test_gan_training_writes_history_grids_and_checkpoint(store, synthetic_mnist):
    cfg = tiny_config("gan")
    ctx, eid = make_ctx(store, cfg)
    progress = []
    ctx.on_progress = progress.append
    result = train_gan(ctx, loaders(synthetic_mnist, cfg)[0])
    assert [r["epoch"] for r in result.history] == [1, 2]
    for key in ("d_loss", "g_loss", "d_real_prob", "d_fake_prob", "epoch_seconds"):
        assert torch.isfinite(torch.tensor(result.history[-1][key]))
    assert set(store.list_epoch_grids(eid)) == {1, 2}
    assert progress and progress[-1]["batch"] == progress[-1]["batches"]
    ckpt = load_checkpoint(ctx.paths.checkpoint)
    assert ckpt["epoch"] == 2 and ckpt["model"] == "gan"


def test_vae_training_tracks_train_and_validation_losses(store, synthetic_mnist):
    cfg = tiny_config("vae")
    ctx, eid = make_ctx(store, cfg)
    train, val = loaders(synthetic_mnist, cfg)
    result = train_vae(ctx, train, val)
    last = result.history[-1]
    for key in ("train_loss", "train_recon", "train_kl", "val_loss", "val_recon", "val_kl"):
        assert torch.isfinite(torch.tensor(last[key])), key
    assert last["train_loss"] == pytest.approx(last["train_recon"] + last["train_kl"], rel=1e-4)
    assert ctx.paths.recon_grid(2).exists()
    # Loss should drop on this tiny set across 2 epochs.
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]


def test_checkpoint_save_load_round_trip(tmp_path):
    cfg = tiny_config("vae")
    vae = VAE(cfg.latent_dim)
    path = tmp_path / "ck" / "last.pt"
    save_checkpoint(path, {"model": "vae", "epoch": 1, "history": [], "train_seconds": 1.0, "config": cfg.to_dict(),
                           "rng_state": torch.get_rng_state(), "state": {"vae": vae.state_dict(), "opt": {}}})
    restored = load_generative_model(load_checkpoint(path), CPU)
    x = torch.rand(2, 1, 28, 28)
    torch.testing.assert_close(restored.decode(vae.encode(x)[0]), vae.eval().decode(vae.encode(x)[0]))
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_fixed_seed_gives_identical_models_on_cpu():
    set_seed(123)
    g1, _ = build_gan(8)
    set_seed(123)
    g2, _ = build_gan(8)
    for a, b in zip(g1.state_dict().values(), g2.state_dict().values()):
        assert torch.equal(a, b)
    g1.eval(), g2.eval()
    z = torch.randn(4, 8, generator=seeded_generator(5))
    assert torch.equal(g1(z), g2(z))


def test_training_is_reproducible_under_fixed_seed(store, synthetic_mnist):
    histories = []
    for _ in range(2):
        cfg = tiny_config("vae", epochs=1)
        set_seed(cfg.seed)
        ctx, _ = make_ctx(store, cfg)
        train, val = loaders(synthetic_mnist, cfg)
        histories.append(train_vae(ctx, train, val).history[0])
    for key in ("train_loss", "val_loss", "val_kl"):
        assert histories[0][key] == histories[1][key]


def test_resume_matches_uninterrupted_run(store, synthetic_mnist):
    cfg = tiny_config("gan", epochs=2)
    set_seed(cfg.seed)
    full_ctx, _ = make_ctx(store, cfg)
    full = train_gan(full_ctx, loaders(synthetic_mnist, cfg)[0]).history

    set_seed(cfg.seed)
    one_epoch = ExperimentConfig.from_dict({**cfg.to_dict(), "epochs": 1})
    ctx, _ = make_ctx(store, one_epoch)
    train_gan(ctx, loaders(synthetic_mnist, one_epoch)[0])
    resume = load_checkpoint(ctx.paths.checkpoint)
    ctx.config = cfg
    resumed = train_gan(ctx, loaders(synthetic_mnist, cfg)[0], resume=resume).history
    assert len(resumed) == 2
    assert resumed[1]["d_loss"] == pytest.approx(full[1]["d_loss"], rel=1e-6)
    assert resumed[1]["g_loss"] == pytest.approx(full[1]["g_loss"], rel=1e-6)


def test_cancellation_raises_and_keeps_last_checkpoint(store, synthetic_mnist):
    cfg = tiny_config("vae", epochs=3)
    calls = {"n": 0}

    def cancel_on_second_epoch():
        calls["n"] += 1
        return calls["n"] > 4  # 4 batches per epoch -> cancel at the start of epoch 2

    ctx, _ = make_ctx(store, cfg, should_cancel=cancel_on_second_epoch)
    train, val = loaders(synthetic_mnist, cfg)
    with pytest.raises(TrainingCancelled):
        train_vae(ctx, train, val)
    assert load_checkpoint(ctx.paths.checkpoint)["epoch"] == 1


def test_divergence_is_detected(store, synthetic_mnist):
    cfg = tiny_config("vae", epochs=1)
    ctx, _ = make_ctx(store, cfg)
    _, val = loaders(synthetic_mnist, cfg)

    class NaNLoader:
        """Yields NaN inputs, which deterministically produce a NaN loss."""
        generator = None

        def __iter__(self):
            for _ in range(2):
                yield torch.full((32, 1, 28, 28), float("nan")), torch.zeros(32, dtype=torch.long)

        def __len__(self):
            return 2

    with pytest.raises(TrainingDivergedError, match="learning_rate"):
        train_vae(ctx, NaNLoader(), val)


def test_generate_images_is_seeded_and_in_unit_range():
    gen = Generator(8).eval()
    a = generate_images(gen, 10, seed=1, device=CPU, batch_size=4)
    b = generate_images(gen, 10, seed=1, device=CPU)
    assert a.shape == (10, 1, 28, 28) and a.min() >= 0 and a.max() <= 1
    assert torch.allclose(a, b, atol=1e-6)
    with pytest.raises(TypeError):
        generate_images(torch.nn.Linear(2, 2), 2, 0, CPU)


def test_evaluate_model_produces_all_metric_groups(synthetic_mnist, tmp_path):
    cfg = tiny_config("gan")
    images, _ = synthetic_mnist
    history = [{"epoch": 1, "d_loss": 1.2, "g_loss": 0.9, "d_real_prob": 0.6, "d_fake_prob": 0.4}]
    m = evaluate_model(Generator(cfg.latent_dim), cfg, history, DigitClassifier(), {"best_val_accuracy": 0.5},
                       images[:200], CPU, samples_dir=tmp_path, log=lambda _: None)
    assert set(m) >= {"objective", "proxy", "distribution", "reference", "evaluation"}
    assert 0 <= m["proxy"]["validity"]["valid_fraction"] <= 1
    assert m["proxy"]["fd_mnist"]["value"] >= 0
    assert m["distribution"]["fid"]["status"] == "skipped"
    assert (tmp_path / "eval_samples.png").exists() and (tmp_path / "real_reference.png").exists()


def test_run_experiment_end_to_end_without_download(store, synthetic_mnist, monkeypatch):
    """Full orchestration with MNIST loading and the evaluator classifier stubbed out."""
    images, labels = synthetic_mnist
    splits = MNISTSplits(images[:128], labels[:128], images[400:600], labels[400:600])
    monkeypatch.setattr(experiment_module, "load_splits", lambda **_: splits)
    monkeypatch.setattr(experiment_module, "get_or_train_classifier",
                        lambda *a, **k: (DigitClassifier().eval(), {"best_val_accuracy": 0.1, "test_accuracy": 0.1}))
    cfg = tiny_config("vae", val_size=200, eval_num_samples=150, fid_num_samples=100)
    eid = store.create(cfg)
    metrics = experiment_module.run_experiment(eid, store, device_preference="cpu")
    assert store.read_status(eid)["state"] == "completed"
    assert metrics["runtime"]["device"] == "cpu" and metrics["runtime"]["train_seconds"] > 0
    meta = store.read_metadata(eid)
    assert meta["config"]["seed"] == cfg.seed and meta["dataset"]["test_set_used_for_training"] is False
    assert meta["environment"]["torch_version"] == torch.__version__
    assert store.paths(eid).metrics_csv.exists() and len(store.read_history(eid)) == 2


def test_run_experiment_records_failure(store, monkeypatch):
    def boom(**_):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(experiment_module, "load_splits", boom)
    eid = store.create(tiny_config("gan"))
    with pytest.raises(RuntimeError):
        experiment_module.run_experiment(eid, store, device_preference="cpu")
    status = store.read_status(eid)
    assert status["state"] == "failed" and "disk on fire" in status["error"]
    assert "disk on fire" in store.paths(eid).log.read_text(encoding="utf-8")


def test_device_selection():
    assert get_device("cpu").type == "cpu"
    assert get_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            get_device("cuda")
    with pytest.raises(ValueError):
        get_device("tpu")
