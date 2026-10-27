import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data.dataset import DatasetUnavailableError, class_distribution, load_mnist_raw, make_train_val_split, mnist_available
from src.data.preprocessing import preprocessing_summary
from src.ui.components import load_mnist_arrays, to_grid_array
from src.utils.config import SPLIT_SEED, data_dir

if not mnist_available():
    st.info(f"MNIST is not on disk yet ({data_dir()}). It is about 12 MB and only needs downloading once.",
            icon=":material/cloud_download:")
    if st.button("Download MNIST", icon=":material/download:", type="primary"):
        try:
            with st.spinner("Downloading MNIST…"):
                load_mnist_raw(train=True)
                load_mnist_raw(train=False)
        except DatasetUnavailableError as exc:
            st.error(str(exc))
        else:
            st.rerun()
    st.stop()

data = load_mnist_arrays()
val_size = st.number_input("Validation split size", min_value=100, max_value=20_000, value=5_000, step=500,
                           help="Images held out from the 60k MNIST training set. All experiments default to 5,000.")
split = make_train_val_split(len(data["train_y"]), int(val_size), SPLIT_SEED)
overlap = np.intersect1d(split.train, split.val).size
sets = {
    "train": (data["train_x"][split.train], data["train_y"][split.train]),
    "validation": (data["train_x"][split.val], data["train_y"][split.val]),
    "test": (data["test_x"], data["test_y"]),
}

with st.container(horizontal=True):
    st.metric("Training split", f"{len(split.train):,}", border=True)
    st.metric("Validation split", f"{len(split.val):,}", border=True)
    st.metric("Official test set", f"{len(data['test_y']):,}", border=True)
    st.metric("Train/val overlap", overlap, help="Computed live from the split indices; must be 0.", border=True)

preview_col, dist_col = st.columns([1, 1.2])
with preview_col, st.container(border=True):
    st.markdown("**Sample preview**")
    which = st.segmented_control("Split", list(sets), default="train", key="ds_split")
    digit = st.pills("Digit", ["all", *range(10)], default="all", key="ds_digit")
    images, labels = sets[which or "train"]
    if digit not in (None, "all"):
        images = images[labels == digit]
    rng = np.random.default_rng(st.session_state.get("ds_seed", 0))
    idx = rng.choice(len(images), size=min(64, len(images)), replace=False)
    st.image(to_grid_array(images[idx] / 255.0), clamp=True, width="stretch")
    if st.button("Shuffle preview", icon=":material/shuffle:"):
        st.session_state.ds_seed = int(rng.integers(1_000_000))
        st.rerun()

with dist_col, st.container(border=True):
    st.markdown("**Class distribution**")
    rows = []
    for name, (_, y) in sets.items():
        counts = class_distribution(y)
        rows += [{"split": name, "digit": str(d), "count": int(c), "share": c / counts.sum()} for d, c in enumerate(counts)]
    dist = pd.DataFrame(rows)
    chart = (
        alt.Chart(dist)
        .mark_bar()
        .encode(
            x=alt.X("digit:N", title="Digit"),
            xOffset="split:N",
            y=alt.Y("share:Q", title="Share of split", axis=alt.Axis(format="%")),
            color=alt.Color("split:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=["split", "digit", "count", alt.Tooltip("share:Q", format=".2%")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart)
    train_x = sets["train"][0]
    st.caption(f"Training-split pixel statistics in [0, 1]: mean {train_x.mean() / 255:.4f}, std {train_x.std() / 255:.4f}. "
               "These are the constants used to standardise classifier input (0.1307 and 0.3081).")

st.subheader("Preprocessing configuration")
pre = pd.DataFrame({m.upper(): preprocessing_summary(m) for m in ("gan", "vae")})
pre["Evaluator classifier"] = ["MNIST uint8 grayscale, 1x28x28", "x / 255 -> [0, 1]", "(x - 0.1307) / 0.3081 inside the model",
                               "none", "[0, 1]"]
st.table(pre)

st.subheader("Train / validation / test separation")
st.markdown(f"""
| Split | Source | Used for | Never used for |
|---|---|---|---|
| **Training** ({len(split.train):,}) | 60k official train, seed-{SPLIT_SEED} permutation | GAN/VAE training, evaluator classifier training | - |
| **Validation** ({len(split.val):,}) | the same permutation, disjoint indices | VAE held-out loss, classifier model selection, real reference for FID / FD-MNIST / diversity | gradient updates |
| **Test** ({len(data['test_y']):,}) | 10k official test | one held-out accuracy report for the evaluator classifier | training, tuning, generative metrics |

`train_limit` takes the first *N* indices of the shuffled training split, so quick experiments use a subset of the
full training data and can never reach validation images. The split seed is fixed and does not change with the
experiment seed, so every experiment is measured against the same reference set.
""")
st.warning("Known MNIST limitations: centred, size-normalised, single-channel digits with clean backgrounds. "
           "Results here do not transfer directly to natural images or to production generative systems.",
           icon=":material/info:")
