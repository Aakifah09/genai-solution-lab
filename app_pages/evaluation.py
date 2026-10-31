import json
import subprocess
import sys
import threading

import altair as alt
import pandas as pd
import streamlit as st

from src.evaluation.classifier import classifier_paths
from src.persistence.storage import read_json
from src.training.jobs import active_job
from src.ui.components import empty_state, experiment_picker, get_store, line_chart
from src.ui.helpers import METRIC_DEFINITIONS, format_duration, format_metric, history_to_long, parse_junit
from src.utils.config import PROJECT_ROOT, test_reports_dir


@st.cache_resource
def _test_run_lock() -> threading.Lock:
    """One test run per server, shared across all browser sessions."""
    return threading.Lock()


store = get_store()

tab_model, tab_tests, tab_defs = st.tabs(["Model evaluation", "Software tests", "Metric definitions"])

with tab_model:
    exp_id = experiment_picker(store, "Evaluated experiment", key="eval_exp", require="metrics_json")
    if exp_id is None:
        empty_state("No evaluated experiments yet. Metrics appear here after a training job completes.",
                    "app_pages/train_models.py", "Train a model")
    else:
        m = store.read_metrics(exp_id)
        config = store.load_config(exp_id)
        history = store.read_history(exp_id)
        proxy, ref, fid = m["proxy"], m["reference"], m["distribution"]["fid"]

        st.markdown("### :material/target: Model objective")
        st.caption("These are the model's own training losses. GAN and VAE losses measure different things and must "
                   "not be compared with each other.")
        if config.model == "gan":
            obj = m["objective"]
            with st.container(horizontal=True):
                st.metric("Final D loss", f"{obj['final_d_loss']:.4f}", border=True)
                st.metric("Final G loss", f"{obj['final_g_loss']:.4f}", border=True)
                st.metric("Mean D(x)", f"{obj['final_d_real_prob']:.3f}", help="Discriminator probability on real images. This is a diagnostic, not an accuracy.", border=True)
                st.metric("Mean D(G(z))", f"{obj['final_d_fake_prob']:.3f}", border=True)
            c1, c2 = st.columns(2)
            c1.altair_chart(line_chart(history_to_long(history, ["d_loss", "g_loss"], {"d_loss": "Discriminator", "g_loss": "Generator"}), "BCE loss"))
            c2.altair_chart(line_chart(history_to_long(history, ["d_real_prob", "d_fake_prob"], {"d_real_prob": "D(x) real", "d_fake_prob": "D(G(z)) fake"}), "Mean D output"))
        else:
            obj = m["objective"]
            with st.container(horizontal=True):
                st.metric("Val reconstruction (nats/image)", f"{obj['final_val_recon']:.2f}", border=True)
                st.metric("Val KL (nats/image)", f"{obj['final_val_kl']:.2f}", border=True)
                st.metric("Val negative ELBO", f"{obj['final_val_neg_elbo']:.2f}", border=True)
                st.metric("Val BCE per pixel", f"{obj['val_recon_per_pixel']:.4f}", border=True)
            st.caption(f"Reduction: {obj['reduction']}. KL weight β = {config.kl_weight}.")
            c1, c2 = st.columns(2)
            c1.altair_chart(line_chart(history_to_long(history, ["train_recon", "val_recon"], {"train_recon": "train", "val_recon": "validation"}), "Reconstruction (nats/image)"))
            c2.altair_chart(line_chart(history_to_long(history, ["train_kl", "val_kl"], {"train_kl": "train", "val_kl": "validation"}), "KL divergence (nats/image)"))

        st.markdown("### :material/psychology: Proxy metrics (held-out classifier)")
        st.caption(f"Computed on {m['evaluation']['num_samples']:,} generated images (sampling seed {m['evaluation']['sample_seed']}) "
                   "and the same number of real validation images. Real-data values show what 'good' looks like.")
        v, rv, d = proxy["validity"], ref["real_validity"], proxy["diversity"]
        with st.container(horizontal=True):
            st.metric(f"Validity rate (conf ≥ {v['threshold']})", f"{v['valid_fraction']:.1%}",
                      f"real data: {rv['valid_fraction']:.1%}", delta_color="off", border=True)
            st.metric("Class entropy (normalised)", f"{v['class_entropy_normalized']:.3f}",
                      f"real data: {rv['class_entropy_normalized']:.3f}", delta_color="off", border=True)
            st.metric("Diversity ratio (features)", f"{d['feature_ratio_to_real']:.3f}", "1.0 = real-data spread", delta_color="off", border=True)
            st.metric("Diversity ratio (pixels)", f"{d['pixel_ratio_to_real']:.3f}", "1.0 = real-data spread", delta_color="off", border=True)
            st.metric("FD-MNIST", f"{proxy['fd_mnist']['value']:.2f}", "lower is better; not standard FID", delta_color="off", border=True)
        counts = pd.DataFrame({"digit": [str(i) for i in range(10)] * 2,
                               "count": v["predicted_class_counts"] + rv["predicted_class_counts"],
                               "set": ["generated"] * 10 + ["real validation"] * 10})
        st.altair_chart(
            alt.Chart(counts).mark_bar().encode(
                x=alt.X("digit:N", title="Predicted digit"), xOffset="set:N", y=alt.Y("count:Q", title="Images"),
                color=alt.Color("set:N", title=None, legend=alt.Legend(orient="bottom")),
                tooltip=["set", "digit", "count"]).properties(height=260, title="Predicted class histogram"))

        st.markdown("### :material/functions: Distribution metric (standard FID)")
        if fid.get("status") == "computed":
            with st.container(horizontal=True):
                st.metric("FID", f"{fid['value']:.2f}", f"{fid['n_generated']:,} vs {fid['n_real']:,} images", delta_color="off", border=True)
                st.metric("FID compute time", format_duration(fid.get("seconds")), border=True)
            st.caption(f"Feature extractor: {fid['feature_extractor']}. Reference: {fid['reference_set']}. "
                       "Compare FID only between runs with the same sample count; it is biased upward at small N.")
        else:
            st.warning(f"FID {fid.get('status')}: {fid.get('reason')}. FD-MNIST above is the labelled alternative.",
                       icon=":material/warning:")

        st.markdown("### :material/timer: Runtime and hardware")
        rt = m["runtime"]
        with st.container(horizontal=True):
            st.metric("Training time", format_duration(rt["train_seconds"]), border=True)
            st.metric("Evaluation time", format_duration(rt["eval_seconds"]), border=True)
            st.metric("Device", rt["device"].upper(), border=True)
        st.caption(rt.get("device_description", ""))

        st.markdown("### :material/verified: Evaluator classifier")
        st.caption("This is the accuracy of the classifier used to evaluate the generative models. It is "
                   "not an accuracy of the GAN or VAE.")
        with st.container(horizontal=True):
            st.metric("Validation accuracy (model selection)", format_metric(ref.get("classifier_val_accuracy"), 4), border=True)
            st.metric("Held-out MNIST test accuracy", format_metric(ref.get("classifier_test_accuracy"), 4), border=True)

        meta = store.read_metadata(exp_id)
        with st.container(horizontal=True):
            st.download_button("Download metrics CSV", store.paths(exp_id).metrics_csv.read_bytes(),
                               file_name=f"{exp_id}_metrics.csv", mime="text/csv", icon=":material/download:")
            curves = store.paths(exp_id).samples / "training_curves.png"
            if curves.exists():
                st.download_button("Download training curves PNG", curves.read_bytes(), file_name=f"{exp_id}_curves.png",
                                   mime="image/png", icon=":material/download:")
            if meta:
                st.download_button("Download metadata JSON", json.dumps(meta, indent=2), file_name=f"{exp_id}_metadata.json",
                                   mime="application/json", icon=":material/download:")
        with st.expander("Raw metrics.json"):
            st.json(m, expanded=False)

with tab_tests:
    st.caption("Automated software tests check that the code behaves correctly: shapes, losses, persistence and "
               "split isolation. They say nothing about how good a model is.")
    report_path = test_reports_dir() / "junit.xml"
    training = active_job(store)
    if st.button("Run tests", icon=":material/play_arrow:", type="primary", disabled=bool(training),
                 help="Paused while a training job runs." if training else "Runs the pytest suite on this server (about a minute)."):
        lock = _test_run_lock()
        if not lock.acquire(blocking=False):
            st.warning("A test run is already in progress. Try again in a minute.", icon=":material/hourglass_top:")
        else:
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                with st.spinner("Running the test suite…"):
                    subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={report_path}"],
                                   cwd=PROJECT_ROOT, capture_output=True, timeout=900)
            except subprocess.TimeoutExpired:
                st.error("The test run took longer than 15 minutes and was stopped.", icon=":material/timer_off:")
            finally:
                lock.release()
    report = parse_junit(report_path)
    if report is None:
        st.info("No test report yet. Click **Run tests** above, or run `python cli.py test` in a terminal.",
                icon=":material/terminal:")
    else:
        with st.container(horizontal=True):
            st.metric("Tests", report["tests"], border=True)
            st.metric("Passed", report["passed"], border=True)
            st.metric("Failed + errors", report["failures"] + report["errors"], border=True)
            st.metric("Skipped", report["skipped"], border=True)
            st.metric("Duration", format_duration(report["seconds"]), border=True)
        st.caption(f"Report timestamp: {report['timestamp']}")
        st.dataframe(pd.DataFrame(report["cases"]), hide_index=True,
                     column_config={"seconds": st.column_config.NumberColumn(format="%.2f")})

with tab_defs:
    st.table(pd.DataFrame(METRIC_DEFINITIONS).set_index("metric"))
    clf = read_json(classifier_paths()[1])
    if clf:
        st.markdown("**Evaluator classifier training history** (training split only; model chosen by validation accuracy)")
        st.dataframe(pd.DataFrame(clf["history"]), hide_index=True)
        if "test_per_class" in clf:
            st.markdown(f"**Evaluator on the held-out MNIST test set** (scikit-learn metrics, {clf['test_size']:,} images): "
                        f"accuracy {clf['test_accuracy']:.4f}, macro-F1 {clf['test_macro_f1']:.4f}")
            c1, c2 = st.columns(2)
            c1.dataframe(pd.DataFrame(clf["test_per_class"]).T.rename_axis("digit"),
                         column_config={k: st.column_config.NumberColumn(format="%.4f") for k in ("precision", "recall", "f1-score")})
            cm = pd.DataFrame(clf["test_confusion_matrix"]).reset_index().melt(id_vars="index", var_name="predicted", value_name="count")
            cm = cm.rename(columns={"index": "true"}).astype({"true": str, "predicted": str})
            c2.altair_chart(alt.Chart(cm).mark_rect().encode(
                x=alt.X("predicted:O", title="Predicted"), y=alt.Y("true:O", title="True"),
                color=alt.Color("count:Q", scale=alt.Scale(type="log", scheme="blues"), legend=None),
                tooltip=["true", "predicted", "count"]).properties(height=300, title="Confusion matrix (log colour scale)"))
