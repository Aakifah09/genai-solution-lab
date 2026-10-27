"""Job management, UI helper functions and headless dashboard rendering."""

import os
import subprocess
import sys

import pandas as pd
import pytest

from src.training import jobs
from src.ui.helpers import (
    comparison_frame,
    config_differences,
    experiment_label,
    fairness_warnings,
    format_duration,
    format_metric,
    history_to_long,
    interpret_comparison,
    parse_junit,
)
from src.utils.config import ExperimentConfig


# -- jobs ------------------------------------------------------------------------

def test_process_liveness():
    assert jobs.is_process_alive(os.getpid())
    assert not jobs.is_process_alive(None)
    assert not jobs.is_process_alive(-5)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not jobs.is_process_alive(proc.pid)


def test_dead_job_is_marked_interrupted(store):
    eid = store.create(ExperimentConfig.for_model("gan"))
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    store.write_job(eid, {"pid": proc.pid})
    store.update_status(eid, state="running", pid=proc.pid)
    assert jobs.active_job(store) is None
    status = store.read_status(eid)
    assert status["state"] == "interrupted" and status["resumable"] is False


def test_live_job_blocks_duplicate_launch(store):
    eid = store.create(ExperimentConfig.for_model("vae"))
    store.write_job(eid, {"pid": os.getpid()})  # this test process is certainly alive
    store.update_status(eid, state="running")
    assert jobs.active_job(store) == eid
    other = store.create(ExperimentConfig.for_model("gan"))
    with pytest.raises(jobs.JobConflictError, match="still running"):
        jobs.launch_job(store, other)


def test_stale_status_pid_does_not_override_new_job_pid(store):
    """After a resume, status.json may still hold the old dead PID; job.json must win."""
    eid = store.create(ExperimentConfig.for_model("vae"))
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    store.update_status(eid, state="queued", pid=proc.pid)
    store.write_job(eid, {"pid": os.getpid()})
    assert jobs.active_job(store) == eid


# -- UI helpers ----------------------------------------------------------------

@pytest.mark.parametrize("seconds, expected", [(None, "-"), (5.25, "5.2s"), (65, "1m 05s"), (3725, "1h 02m")])
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_format_metric():
    assert format_metric(None) == "n/a"
    assert format_metric(float("nan")) == "n/a"
    assert format_metric(0.12345, 2) == "0.12"
    assert format_metric(3) == "3"


def test_experiment_label():
    label = experiment_label({"experiment_id": "gan-20261026-204100-abcdef", "model": "gan", "epochs": 5,
                              "train_size": 55000, "state": "completed"})
    assert label == "GAN · 2026-10-26 20:41 · 5 ep · 55,000 imgs · completed"


def test_history_to_long():
    hist = pd.DataFrame({"epoch": [1, 2], "d_loss": [1.0, 0.9], "g_loss": [2.0, 1.5], "other": [0, 0]})
    long = history_to_long(hist, ["d_loss", "g_loss", "missing"], {"d_loss": "D"})
    assert len(long) == 4 and set(long["series"]) == {"D", "g_loss"}
    assert history_to_long(pd.DataFrame(), ["x"]).empty


def test_config_differences_and_fairness():
    a = ExperimentConfig.for_model("gan", epochs=5).to_dict()
    b = ExperimentConfig.for_model("vae", epochs=10).to_dict()
    diffs = config_differences({"gan-1": a, "vae-1": b})
    assert "epochs" in diffs and "learning_rate" in diffs and "model" not in diffs and "seed" not in diffs
    warnings = fairness_warnings({"gan-20260101-000000-aaaaaa": a, "vae-20260101-000000-bbbbbb": b})
    assert len(warnings) == 1 and "epochs" in warnings[0]


def test_comparison_and_interpretation_respect_metric_direction():
    rows = [
        {"experiment_id": "A", "validity_rate": 0.9, "fd_mnist": 5.0, "diversity_feature_ratio": 0.7, "train_seconds": 100.0},
        {"experiment_id": "B", "validity_rate": 0.6, "fd_mnist": 2.0, "diversity_feature_ratio": 0.95, "train_seconds": 50.0},
    ]
    table = comparison_frame(rows)
    assert table.set_index("metric").loc["fd_mnist", "better"] == "lower"
    text = " ".join(interpret_comparison(rows, {"A": "GAN", "B": "VAE"}))
    assert "**validity_rate** (higher is better): GAN = 0.9" in text
    assert "**fd_mnist** (lower is better): VAE = 2" in text
    assert "**diversity_feature_ratio** (closer to 1 is better): VAE = 0.95" in text
    assert "fastest was VAE" in text
    assert interpret_comparison(rows[:1], {}) == []


def test_parse_junit(tmp_path):
    xml = tmp_path / "junit.xml"
    xml.write_text("""<?xml version="1.0"?><testsuites><testsuite name="pytest" errors="0" failures="1" skipped="1"
        tests="3" time="1.5" timestamp="2026-10-26T20:00:00"><testcase classname="t" name="a" time="0.1"/>
        <testcase classname="t" name="b" time="0.2"><failure message="x"/></testcase>
        <testcase classname="t" name="c" time="0"><skipped/></testcase></testsuite></testsuites>""")
    report = parse_junit(xml)
    assert report["tests"] == 3 and report["passed"] == 1 and report["failures"] == 1
    assert [c["outcome"] for c in report["cases"]] == ["passed", "failed", "skipped"]
    assert parse_junit(tmp_path / "missing.xml") is None


# -- headless dashboard ----------------------------------------------------------

PAGES = ["app_pages/overview.py", "app_pages/dataset_explorer.py", "app_pages/train_models.py",
         "app_pages/generated_samples.py", "app_pages/evaluation.py", "app_pages/comparison.py",
         "app_pages/architecture.py"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exceptions_on_empty_workspace(page):
    from streamlit.testing.v1 import AppTest

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
