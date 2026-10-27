import streamlit as st

from src.persistence.storage import StorageError
from src.training.jobs import JobConflictError, active_job, force_stop, launch_job, request_cancel
from src.ui.components import get_store, state_text
from src.ui.helpers import experiment_label, format_duration
from src.utils.config import ConfigError, ExperimentConfig

store = get_store()

PRESETS = {
    "Quick demo": {"epochs": 3, "train_limit": 10_000, "eval_num_samples": 1_000, "fid_num_samples": 500, "compute_fid": False},
    "Standard": {"epochs": 10, "train_limit": None, "eval_num_samples": 2_000, "fid_num_samples": 1_000, "compute_fid": True},
    "Long": {"epochs": 25, "train_limit": None, "eval_num_samples": 2_000, "fid_num_samples": 1_000, "compute_fid": True},
}
PRESET_HELP = ("Quick demo trains on 10k images for 3 epochs: about 1 min (VAE) to 3 min (GAN) of CPU training, plus evaluation. "
               "Standard and Long use all 55k training images; on the development laptop CPU a full GAN epoch measured "
               "4–15 min depending on background load, and a VAE epoch about 1–5 min.")


def tail(path, lines: int = 25) -> str:
    if not path.exists():
        return "(no log yet)"
    with open(path, encoding="utf-8", errors="replace") as fh:
        return "".join(fh.readlines()[-lines:])


@st.fragment(run_every="2s")
def job_monitor() -> None:
    running = active_job(store)
    if running is None:
        if st.session_state.get("monitoring"):
            st.session_state.monitoring = None
            st.rerun()  # full rerun: re-enable the form and refresh the experiment list
        return
    st.session_state.monitoring = running
    status = store.read_status(running)
    paths = store.paths(running)
    state = status.get("state", "queued")
    with st.container(border=True):
        st.markdown(f"**Active job** `{running}` · {state_text(state)}")
        epoch, epochs = status.get("epoch") or 0, status.get("epochs") or 1
        batch, batches = status.get("batch") or 0, status.get("batches") or 1
        done = min(1.0, ((max(epoch, 1) - 1) + batch / batches) / epochs) if epoch else 0.0
        if state == "evaluating":
            done = 1.0
        st.progress(done, text=f"Epoch {epoch}/{epochs} · batch {batch}/{batches}" if state == "running" else state)
        with st.container(horizontal=True):
            st.metric("Elapsed (training)", format_duration(status.get("elapsed_seconds")), border=True)
            for name, value in (status.get("losses") or {}).items():
                st.metric(name.replace("_", " "), f"{value:.4f}", border=True)
            st.metric("Device", status.get("device", "-"), border=True)
        if state == "queued":
            st.caption("Starting process and loading data…")
        if state == "evaluating":
            st.info("Training finished. The evaluation step generates samples, runs the held-out classifier and "
                    "computes FD-MNIST, then FID if enabled. FID is slow on CPU.", icon=":material/hourglass_top:")
        grids = store.list_epoch_grids(running)
        if grids:
            last_epoch, grid = list(grids.items())[-1]
            st.image(str(grid), caption=f"Fixed-noise samples after epoch {last_epoch}", width=360)
        if store.cancel_requested(running):
            st.warning("Cancellation requested. The job stops at the next batch and keeps the last completed epoch.")
        with st.container(horizontal=True):
            if st.button("Cancel training", icon=":material/stop_circle:", disabled=store.cancel_requested(running)):
                request_cancel(store, running)
                st.rerun(scope="fragment")
            if st.button("Force stop", icon=":material/dangerous:",
                         help="Terminate the process immediately. Use only if cancellation does not respond."):
                force_stop(store, running)
                st.rerun()
        with st.expander("Live log"):
            st.code(tail(paths.log), language="log")


job_monitor()
running = active_job(store)

st.subheader("New experiment")
if running:
    st.info("Only one training job runs at a time; this keeps CPU/GPU use and results predictable. "
            "The form unlocks when the active job finishes.", icon=":material/lock:")

top = st.container(horizontal=True)
with top:
    model = st.segmented_control("Model", ["gan", "vae"], format_func=str.upper, default="gan", key="train_model",
                                 required=True)
    preset = st.segmented_control("Preset", list(PRESETS), default="Quick demo", key="train_preset", help=PRESET_HELP,
                                  required=True)
base = ExperimentConfig.for_model(model, **{k: v for k, v in PRESETS[preset].items()})
k = f"{model}_{preset}"  # new widget keys per model/preset so defaults update

with st.form(f"train_form_{k}", border=True):
    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input("Epochs", 1, 500, base.epochs, key=f"ep_{k}")
    batch_size = c2.number_input("Batch size", 2, 4096, base.batch_size, key=f"bs_{k}")
    latent_dim = c3.number_input("Latent dimension", 2, 1024, base.latent_dim, key=f"ld_{k}")
    c1, c2, c3 = st.columns(3)
    lr = c1.number_input("Learning rate", 1e-6, 1.0, base.learning_rate, format="%.5f", step=1e-4, key=f"lr_{k}")
    seed = c2.number_input("Seed", 0, 2**32 - 1, base.seed, key=f"seed_{k}")
    full = c3.checkbox("Use the full training split (55,000)", base.train_limit is None, key=f"full_{k}")
    c1, c2, c3 = st.columns(3)
    train_limit = c1.number_input("Training images (when not full)", 1, 55_000, base.train_limit or 10_000, step=1_000, key=f"tl_{k}")
    if model == "gan":
        beta1 = c2.number_input("Adam β1", 0.0, 0.999, base.beta1, key=f"b1_{k}")
        smoothing = c3.number_input("Label smoothing", 0.0, 0.49, base.label_smoothing, key=f"ls_{k}",
                                    help="Real targets become 1 - smoothing; this reduces discriminator over-confidence.")
        extra = {"beta1": beta1, "label_smoothing": smoothing}
    else:
        kl_weight = c2.number_input("KL weight (β)", 0.0, 100.0, base.kl_weight, key=f"kl_{k}",
                                    help="1.0 gives the standard ELBO. Higher values trade reconstruction for a smoother latent space.")
        extra = {"kl_weight": kl_weight}
    st.markdown("**Evaluation**")
    c1, c2, c3, c4 = st.columns(4)
    eval_n = c1.number_input("Evaluation samples", 100, 10_000, base.eval_num_samples, step=100, key=f"en_{k}")
    threshold = c2.number_input("Validity threshold", 0.5, 0.999, base.validity_threshold, key=f"vt_{k}")
    fid = c3.checkbox("Compute standard FID (slow on CPU)", base.compute_fid, key=f"fid_{k}")
    fid_n = c4.number_input("FID samples", 100, 10_000, base.fid_num_samples, step=100, key=f"fn_{k}")
    submitted = st.form_submit_button("Start training", type="primary", icon=":material/rocket_launch:", disabled=bool(running))

if submitted:
    try:
        config = ExperimentConfig.for_model(
            model, epochs=int(epochs), batch_size=int(batch_size), latent_dim=int(latent_dim), learning_rate=float(lr),
            seed=int(seed), train_limit=None if full else int(train_limit), eval_num_samples=int(eval_n),
            validity_threshold=float(threshold), compute_fid=bool(fid), fid_num_samples=int(fid_n), **extra,
        )
    except ConfigError as exc:
        st.error("The configuration is invalid:\n\n" + "\n".join(f"- {p}" for p in exc.problems), icon=":material/error:")
    else:
        try:
            exp_id = store.create(config)
            launch_job(store, exp_id)
        except JobConflictError as exc:
            st.error(str(exc), icon=":material/lock:")
        except (OSError, StorageError) as exc:
            st.error(f"Could not start the job: {exc}", icon=":material/error:")
        else:
            st.session_state.monitoring = exp_id
            st.toast(f"Started {exp_id}", icon=":material/rocket_launch:")
            st.rerun()

st.subheader("Recovery")
summaries = store.summaries()
problem_rows = [] if summaries.empty else summaries[summaries["state"].isin(["failed", "cancelled", "interrupted"])].to_dict("records")
if not problem_rows:
    st.caption("No failed, cancelled or interrupted experiments.")
for row in problem_rows:
    exp_id = row["experiment_id"]
    status = store.read_status(exp_id)
    resumable = store.paths(exp_id).checkpoint.exists() and status.get("resumable", True)
    with st.container(border=True):
        st.markdown(f"{state_text(row['state'])} · {experiment_label(row)}")
        if status.get("error"):
            st.code(status["error"], language=None)
        if status.get("hint"):
            st.caption(status["hint"])
        with st.container(horizontal=True):
            if st.button("Resume from checkpoint", key=f"resume_{exp_id}", icon=":material/replay:",
                         disabled=bool(running) or not resumable,
                         help=None if resumable else "No usable checkpoint: the run stopped before the first epoch finished, or training diverged."):
                try:
                    launch_job(store, exp_id, resume=True)
                except JobConflictError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.monitoring = exp_id
                    st.rerun()
            if st.button("Delete", key=f"del_{exp_id}", icon=":material/delete:"):
                st.session_state[f"confirm_del_{exp_id}"] = True
            if st.session_state.get(f"confirm_del_{exp_id}"):
                if st.button("Confirm delete", key=f"confirm_{exp_id}", type="primary", icon=":material/delete_forever:"):
                    store.delete(exp_id)
                    st.rerun()
        with st.expander("Log tail"):
            st.code(tail(store.paths(exp_id).log), language="log")

st.subheader("All experiments")
if summaries.empty:
    st.caption("No experiments yet.")
else:
    cols = [c for c in ["experiment_id", "model", "state", "epochs", "train_size", "batch_size", "latent_dim",
                        "learning_rate", "seed", "validity_rate", "fd_mnist", "fid", "train_seconds"] if c in summaries]
    st.dataframe(summaries[cols], hide_index=True,
                 column_config={"train_seconds": st.column_config.NumberColumn("train s", format="%.0f"),
                                "validity_rate": st.column_config.NumberColumn(format="%.3f"),
                                "fd_mnist": st.column_config.NumberColumn(format="%.2f"),
                                "fid": st.column_config.NumberColumn(format="%.2f")})
