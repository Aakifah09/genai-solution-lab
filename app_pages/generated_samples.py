import io

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.models.gan import Generator
from src.ui.components import empty_state, experiment_picker, get_store, load_sampling_model, to_grid_array
from src.ui.helpers import experiment_label
from src.utils.reproducibility import seeded_generator


def png_bytes(array: np.ndarray) -> bytes:
    img = Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8).squeeze())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


store = get_store()
exp_id = experiment_picker(store, "Experiment", key="samples_exp")
if exp_id is None:
    empty_state("No experiments with samples yet.", "app_pages/train_models.py", "Train a model")
    st.stop()

paths = store.paths(exp_id)
config = store.load_config(exp_id)
grids = store.list_epoch_grids(exp_id)

tab_epochs, tab_real, tab_live, tab_compare = st.tabs(
    ["Across epochs", "Real vs generated", "Generate live", "Across experiments"])

with tab_epochs:
    if not grids:
        st.caption("No epoch has finished yet.")
    else:
        epochs = list(grids)
        epoch = st.select_slider("Epoch", epochs, value=epochs[-1]) if len(epochs) > 1 else epochs[0]
        cols = st.columns(2 if config.model == "vae" else 1)
        with cols[0]:
            st.image(str(grids[epoch]), caption=f"Samples from fixed latent noise after epoch {epoch}", width=420)
            st.download_button("Download PNG", grids[epoch].read_bytes(), file_name=f"{exp_id}_epoch_{epoch:03d}.png",
                               mime="image/png", icon=":material/download:", key="dl_epoch")
        if config.model == "vae" and paths.recon_grid(epoch).exists():
            with cols[1]:
                st.image(str(paths.recon_grid(epoch)), width=420,
                         caption="Reconstructions of held-out validation images: odd rows are originals, even rows decode(μ)")
                st.download_button("Download PNG", paths.recon_grid(epoch).read_bytes(),
                                   file_name=f"{exp_id}_recon_{epoch:03d}.png", mime="image/png",
                                   icon=":material/download:", key="dl_recon")
        st.caption("The latent vectors are the same for every epoch, so you can watch the same points evolve.")

with tab_real:
    real, gen = paths.samples / "real_reference.png", paths.samples / "eval_samples.png"
    if not (real.exists() and gen.exists()):
        st.caption("Available once the experiment has been evaluated.")
    else:
        c1, c2 = st.columns(2)
        c1.image(str(real), caption="Real MNIST validation images (reference set)", width=420)
        c2.image(str(gen), caption=f"Generated evaluation samples (seed {config.seed + 10_000})", width=420)
        c2.download_button("Download generated grid", gen.read_bytes(), file_name=f"{exp_id}_eval_samples.png",
                           mime="image/png", icon=":material/download:")

with tab_live:
    if not paths.checkpoint.exists():
        st.caption("No checkpoint yet.")
    else:
        model = load_sampling_model(str(paths.checkpoint), paths.checkpoint.stat().st_mtime)
        c1, c2 = st.columns([1, 2])
        with c1:
            seed = st.number_input("Sampling seed", 0, 2**31 - 1, 0, key="live_seed")
            mode = st.segmented_control("Mode", ["Random samples", "Latent interpolation"], default="Random samples",
                                        key="live_mode", required=True)
        rng = seeded_generator(int(seed))
        with torch.no_grad():
            if mode == "Random samples":
                z = torch.randn(64, model.latent_dim, generator=rng)
            else:
                # 8 rows, each a straight line between two random latent points.
                a, b = torch.randn(8, model.latent_dim, generator=rng), torch.randn(8, model.latent_dim, generator=rng)
                t = torch.linspace(0, 1, 8).view(1, 8, 1)
                z = (a.unsqueeze(1) * (1 - t) + b.unsqueeze(1) * t).reshape(64, -1)
            images = (model(z) + 1) / 2 if isinstance(model, Generator) else model.decode(z)
        grid = to_grid_array(images.clamp(0, 1))
        with c2:
            st.image(grid, clamp=True, width=420,
                     caption="Generated now from the saved checkpoint" + (" (each row interpolates between two latent points)" if mode != "Random samples" else ""))
        c1.download_button("Download PNG", png_bytes(grid), file_name=f"{exp_id}_seed{seed}.png", mime="image/png",
                           icon=":material/download:")

with tab_compare:
    summaries = store.summaries()
    rows = summaries.to_dict("records")
    with_grids = [r for r in rows if store.list_epoch_grids(r["experiment_id"])]
    labels = {r["experiment_id"]: experiment_label(r) for r in with_grids}
    chosen = st.multiselect("Experiments", list(labels), default=list(labels)[:3], format_func=labels.get, max_selections=4)
    if chosen:
        cols = st.columns(len(chosen))
        for col, eid in zip(cols, chosen):
            g = store.list_epoch_grids(eid)
            last = list(g)[-1]
            col.image(str(g[last]), caption=f"{labels[eid]} · epoch {last}", width="stretch")
