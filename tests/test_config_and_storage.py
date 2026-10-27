import json

import pandas as pd
import pytest

from src.persistence.storage import (
    ExperimentNotFoundError,
    StorageError,
    atomic_write_json,
    metrics_to_frame,
    new_experiment_id,
    read_json,
    validate_experiment_id,
)
from src.utils.config import ConfigError, ExperimentConfig


# -- configuration -----------------------------------------------------------

def test_model_specific_defaults():
    gan, vae = ExperimentConfig.for_model("gan"), ExperimentConfig.for_model("vae")
    assert gan.learning_rate == 2e-4 and gan.beta1 == 0.5
    assert vae.learning_rate == 1e-3 and vae.latent_dim == 16
    assert gan.effective_train_size == 55_000


@pytest.mark.parametrize("overrides, fragment", [
    ({"epochs": 0}, "epochs"),
    ({"batch_size": 1}, "batch_size"),
    ({"learning_rate": 0.0}, "learning_rate"),
    ({"learning_rate": -1e-3}, "learning_rate"),
    ({"latent_dim": 1}, "latent_dim"),
    ({"epochs": 2.5}, "epochs must be an integer"),
    ({"epochs": True}, "epochs must be an integer"),
    ({"train_limit": 60_000}, "exceeds the training split"),
    ({"train_limit": 50, "batch_size": 128}, "at least one full batch"),
    ({"validity_threshold": 1.0}, "validity_threshold"),
    ({"eval_num_samples": 6000}, "cannot exceed val_size"),
    ({"fid_num_samples": 1500, "eval_num_samples": 1000}, "fid_num_samples"),
    ({"label_smoothing": 0.6}, "label_smoothing"),
    ({"compute_fid": "yes"}, "compute_fid"),
])
def test_invalid_configs_raise_clear_errors(overrides, fragment):
    with pytest.raises(ConfigError) as info:
        ExperimentConfig.for_model("gan", **overrides)
    assert any(fragment in p for p in info.value.problems), info.value.problems


def test_multiple_problems_reported_together():
    with pytest.raises(ConfigError) as info:
        ExperimentConfig.for_model("vae", epochs=0, batch_size=0)
    assert len(info.value.problems) == 2


def test_unknown_model_and_unknown_keys_rejected():
    with pytest.raises(ConfigError):
        ExperimentConfig.for_model("diffusion")
    with pytest.raises(ConfigError, match="unknown configuration keys"):
        ExperimentConfig.from_dict({"model": "gan", "dropout": 0.1})


def test_config_round_trip():
    cfg = ExperimentConfig.for_model("vae", epochs=3, train_limit=1000, kl_weight=2.0)
    assert ExperimentConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


# -- persistence -------------------------------------------------------------

def test_experiment_ids_are_validated():
    eid = new_experiment_id("gan")
    assert validate_experiment_id(eid) == eid
    for bad in ["../evil", "gan-1-2-3", "vae-20260101-120000-XYZ123", ""]:
        with pytest.raises(StorageError):
            validate_experiment_id(bad)


def test_atomic_json_round_trip_and_corruption(tmp_path):
    path = tmp_path / "a" / "b.json"
    atomic_write_json(path, {"x": 1, "y": [1.5, None]})
    assert read_json(path) == {"x": 1, "y": [1.5, None]}
    assert list(path.parent.glob("*.tmp")) == []
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(StorageError, match="corrupted"):
        read_json(path)
    assert read_json(tmp_path / "missing.json", default={}) == {}


def test_store_create_and_status_lifecycle(store):
    cfg = ExperimentConfig.for_model("gan", epochs=2)
    eid = store.create(cfg)
    paths = store.paths(eid)
    assert paths.samples.is_dir() and paths.checkpoints.is_dir()
    assert store.load_config(eid) == cfg
    assert store.read_status(eid)["state"] == "queued"
    store.update_status(eid, state="running", epoch=1)
    status = store.read_status(eid)
    assert status["state"] == "running" and status["epoch"] == 1 and "updated_at" in status
    assert store.experiment_ids() == [eid]
    with pytest.raises(ExperimentNotFoundError):
        store.paths("gan-20990101-000000-abcdef")


def test_cancel_flag(store):
    eid = store.create(ExperimentConfig.for_model("vae"))
    assert not store.cancel_requested(eid)
    store.request_cancel(eid)
    assert store.cancel_requested(eid)
    store.clear_cancel(eid)
    assert not store.cancel_requested(eid)


def test_history_metrics_and_metadata_persistence(store):
    eid = store.create(ExperimentConfig.for_model("vae"))
    history = [{"epoch": 1, "train_loss": 120.5}, {"epoch": 2, "train_loss": 110.25}]
    store.write_history(eid, history)
    pd.testing.assert_frame_equal(store.read_history(eid), pd.DataFrame(history))
    metrics = {"model": "vae", "proxy": {"validity": {"valid_fraction": 0.8, "predicted_class_counts": [1, 2]},
                                          "fd_mnist": {"value": 3.5}},
               "distribution": {"fid": {"value": None, "status": "skipped"}}, "runtime": {"train_seconds": 12.0}}
    store.write_metrics(eid, metrics)
    assert store.read_metrics(eid) == metrics
    csv = pd.read_csv(store.paths(eid).metrics_csv)
    assert set(csv.columns) == {"experiment_id", "category", "metric", "value"}
    assert csv.loc[csv.metric == "proxy.validity.valid_fraction", "value"].astype(float).item() == 0.8
    store.write_metadata(eid, {"seed": 42})
    assert store.read_metadata(eid) == {"seed": 42}
    row = store.summary(eid)
    assert row["validity_rate"] == 0.8 and row["fd_mnist"] == 3.5 and row["fid"] is None


def test_metrics_to_frame_flattens_lists():
    df = metrics_to_frame("gan-20260101-000000-abcdef", {"a": {"b": [1, 2]}, "c": 3})
    assert df.set_index("metric").loc["a.b", "value"] == "[1, 2]"
    assert df.set_index("metric").loc["c", "category"] == "c"


def test_delete_refuses_active_experiment(store):
    eid = store.create(ExperimentConfig.for_model("gan"))
    store.update_status(eid, state="running")
    with pytest.raises(StorageError, match="still active"):
        store.delete(eid)
    store.update_status(eid, state="failed")
    store.delete(eid)
    assert not store.exists(eid)
