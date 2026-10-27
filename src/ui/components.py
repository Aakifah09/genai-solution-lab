"""Streamlit-specific shared components and cached loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.persistence.storage import ExperimentStore
from src.ui.helpers import STATE_ICONS, experiment_label


def get_store() -> ExperimentStore:
    return ExperimentStore()


@st.cache_resource
def device_info() -> dict[str, Any]:
    from src.utils.reproducibility import environment_info, get_device

    return environment_info(get_device())


@st.cache_data(show_spinner="Loading MNIST from disk…", max_entries=2)
def load_mnist_arrays() -> dict[str, np.ndarray]:
    """Raw uint8 MNIST arrays (local files only; downloading is an explicit user action)."""
    from src.data.dataset import load_mnist_raw

    tr_x, tr_y = load_mnist_raw(train=True, download=False)
    te_x, te_y = load_mnist_raw(train=False, download=False)
    return {"train_x": tr_x.numpy(), "train_y": tr_y.numpy(), "test_x": te_x.numpy(), "test_y": te_y.numpy()}


@st.cache_resource(max_entries=2, show_spinner="Loading checkpoint…")
def load_sampling_model(checkpoint_path: str, mtime: float) -> torch.nn.Module:
    """Generator/VAE for interactive sampling. Keyed on mtime so a newer checkpoint reloads;
    max_entries=2 avoids keeping many model copies in memory."""
    from src.training.trainer import load_checkpoint, load_generative_model

    return load_generative_model(load_checkpoint(Path(checkpoint_path)), torch.device("cpu"))


def to_grid_array(images01: np.ndarray | torch.Tensor, nrow: int = 8) -> np.ndarray:
    from torchvision.utils import make_grid

    t = torch.as_tensor(images01, dtype=torch.float32)
    if t.dim() == 3:
        t = t.unsqueeze(1)
    grid = make_grid(t, nrow=nrow, padding=2, pad_value=0.25)
    return grid.permute(1, 2, 0).numpy()


def line_chart(df_long: pd.DataFrame, y_title: str, height: int = 280) -> alt.Chart:
    return (
        alt.Chart(df_long)
        .mark_line(point=alt.OverlayMarkDef(size=30))
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=["series:N", "epoch:Q", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=height)
    )


def experiment_picker(store: ExperimentStore, label: str, states: tuple[str, ...] | None = None, key: str | None = None,
                      require: str | None = None) -> str | None:
    """Selectbox over experiments filtered by state and/or by an existing file attribute of ExperimentPaths."""
    summaries = store.summaries()
    if summaries.empty:
        return None
    rows = summaries.to_dict("records")
    if states:
        rows = [r for r in rows if r.get("state") in states]
    if require:
        rows = [r for r in rows if getattr(store.paths(r["experiment_id"]), require).exists()]
    if not rows:
        return None
    labels = {r["experiment_id"]: experiment_label(r) for r in rows}
    return st.selectbox(label, list(labels), format_func=labels.get, key=key)


def state_text(state: str | None) -> str:
    state = state or "unknown"
    return f"{STATE_ICONS.get(state, STATE_ICONS['unknown'])} {state}"


def empty_state(message: str, page: str | None = None, page_label: str | None = None) -> None:
    st.info(message, icon=":material/info:")
    if page:
        st.page_link(page, label=page_label or "Go", icon=":material/arrow_forward:")
