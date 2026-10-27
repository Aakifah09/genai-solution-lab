import altair as alt
import pandas as pd
import streamlit as st

from src.ui.components import empty_state, get_store, line_chart
from src.ui.helpers import (
    comparison_frame,
    config_differences,
    experiment_label,
    fairness_warnings,
    format_duration,
    history_to_long,
    interpret_comparison,
)

store = get_store()
summaries = store.summaries()
done = [] if summaries.empty else summaries[summaries["state"] == "completed"].to_dict("records")
if len(done) < 2:
    empty_state("At least two completed experiments are needed for a comparison.", "app_pages/train_models.py", "Train models")
    st.stop()

labels = {r["experiment_id"]: experiment_label(r) for r in done}
# Default: the latest completed GAN and VAE.
defaults = []
for model in ("gan", "vae"):
    latest = next((r["experiment_id"] for r in done if r["model"] == model), None)
    if latest:
        defaults.append(latest)
chosen = st.multiselect("Experiments to compare", list(labels), default=defaults or list(labels)[:2],
                        format_func=labels.get, max_selections=6)
if len(chosen) < 2:
    st.caption("Select at least two experiments.")
    st.stop()

rows = [r for r in done if r["experiment_id"] in chosen]
configs = {eid: store.load_config(eid).to_dict() for eid in chosen}
short = {eid: f"{eid.split('-')[0].upper()}…{eid[-6:]}" for eid in chosen}

warnings = fairness_warnings(configs)
if warnings:
    st.warning("These settings differ, so metric differences are partly caused by setup rather than by model "
               "family:\n\n" + "\n".join(f"- {w}" for w in warnings), icon=":material/balance:")

st.subheader("Configuration")
cfg_df = pd.DataFrame(configs).rename(columns=short)
diffs = set(config_differences(configs))
cfg_df.insert(0, "differs", ["yes" if k in diffs else "" for k in cfg_df.index])
st.dataframe(cfg_df.astype(str), column_config={"differs": st.column_config.TextColumn(width="small")})

st.subheader("Comparable metrics")
st.caption("These metrics use an identical procedure for every model: the same classifier, the same real "
           "validation reference and the same formula. Model-specific losses are shown separately below.")
table = comparison_frame(rows).rename(columns=short)
st.dataframe(table, hide_index=True)
for line in interpret_comparison(rows, short):
    st.markdown(f"- {line}")
st.caption("These statements only restate the measured numbers. Differences between single runs can fall within "
           "seed-to-seed variation; repeat with several seeds before drawing conclusions.")

metric_rows = table.melt(id_vars=["metric", "better"], var_name="experiment", value_name="value").dropna()
metric_rows = metric_rows[~metric_rows["metric"].isin(["train_seconds", "eval_seconds"])]
metric_rows["value"] = metric_rows["value"].astype(float)
st.altair_chart(
    alt.Chart(metric_rows).mark_bar().encode(
        x=alt.X("experiment:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("value:Q", title=None),
        color=alt.Color("experiment:N", legend=None), tooltip=["metric", "experiment", alt.Tooltip("value:Q", format=".4f")],
    ).properties(height=160, width=150).facet(facet=alt.Facet("metric:N", title=None), columns=4).resolve_scale(y="independent"))

st.subheader("Training curves")
st.caption("GAN and VAE losses are plotted separately because they are different objectives on different scales.")
gan_ids = [e for e in chosen if e.startswith("gan")]
vae_ids = [e for e in chosen if e.startswith("vae")]


def combined(ids, cols, names):
    frames = []
    for eid in ids:
        long = history_to_long(store.read_history(eid), cols, names)
        long["series"] = short[eid] + " · " + long["series"]
        frames.append(long)
    return pd.concat(frames) if frames else pd.DataFrame(columns=["epoch", "series", "value"])


c1, c2 = st.columns(2)
if gan_ids:
    c1.markdown("**GAN: generator and discriminator loss**")
    c1.altair_chart(line_chart(combined(gan_ids, ["d_loss", "g_loss"], {"d_loss": "D", "g_loss": "G"}), "BCE loss"))
if vae_ids:
    c2.markdown("**VAE: validation reconstruction and KL**")
    c2.altair_chart(line_chart(combined(vae_ids, ["val_recon", "val_kl"], {"val_recon": "recon", "val_kl": "KL"}), "nats / image"))

st.subheader("Samples")
cols = st.columns(len(chosen))
for col, eid in zip(cols, chosen):
    grid = store.paths(eid).samples / "eval_samples.png"
    if grid.exists():
        col.image(str(grid), caption=short[eid], width="stretch")

st.subheader("Runtime and hardware")
rt = []
for eid in chosen:
    meta = store.read_metadata(eid) or {}
    m = store.read_metrics(eid) or {}
    runtime = m.get("runtime", {})
    rt.append({"experiment": short[eid], "train": format_duration(runtime.get("train_seconds")),
               "evaluate": format_duration(runtime.get("eval_seconds")),
               "device": runtime.get("device_description"), "torch": meta.get("environment", {}).get("torch_version")})
st.dataframe(pd.DataFrame(rt), hide_index=True)

export = pd.concat([pd.read_csv(store.paths(e).metrics_csv) for e in chosen if store.paths(e).metrics_csv.exists()])
st.download_button("Download comparison CSV", export.to_csv(index=False), file_name="comparison_metrics.csv",
                   mime="text/csv", icon=":material/download:")
