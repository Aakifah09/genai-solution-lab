import streamlit as st

from src.data.dataset import mnist_available
from src.evaluation.classifier import classifier_paths
from src.models.gan import Discriminator, Generator
from src.models.vae import VAE
from src.persistence.storage import read_json
from src.ui.components import device_info, empty_state, get_store, state_text
from src.ui.helpers import ARCHITECTURE_MERMAID, experiment_label, format_duration, format_metric
from src.utils.config import ExperimentConfig


@st.cache_resource
def parameter_counts() -> dict[str, int]:
    gan, vae = ExperimentConfig.for_model("gan"), ExperimentConfig.for_model("vae")
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    return {"generator": count(Generator(gan.latent_dim)), "discriminator": count(Discriminator()),
            "vae": count(VAE(vae.latent_dim))}


st.markdown(
    "An end-to-end lab for **training, evaluating and comparing two generative model families**, "
    "a GAN and a VAE, on MNIST. It covers reproducible configuration, leakage-free data splits, "
    "background training jobs, held-out evaluation with clearly labelled proxy metrics, "
    "and persistent experiment tracking."
)
st.caption("Small research models trained from scratch for learning and comparison. They are not pretrained "
           "foundation models and not comparable to LLMs.")

store = get_store()
summaries = store.summaries()
info = device_info()
clf_report = read_json(classifier_paths()[1])

with st.container(horizontal=True):
    st.metric("Experiments", 0 if summaries.empty else len(summaries), border=True)
    st.metric("Completed", 0 if summaries.empty else int((summaries["state"] == "completed").sum()), border=True)
    st.metric("Device", "CUDA" if info["cuda_available"] else "CPU", border=True)
    st.metric("MNIST on disk", "yes" if mnist_available() else "no", border=True)
    st.metric("Evaluator test accuracy",
              f"{clf_report['test_accuracy']:.2%}" if clf_report else "not trained",
              help="Accuracy of the held-out digit classifier on the MNIST test set. This is the evaluator's "
                   "accuracy, not the GAN's or VAE's.", border=True)

st.subheader("Model cards")
counts = parameter_counts()
gan_col, vae_col = st.columns(2)
with gan_col, st.container(border=True):
    st.markdown("#### :material/swap_horiz: GAN (DCGAN-style)")
    st.markdown(
        f"- **Generator** ({counts['generator']:,} params): z ∈ ℝ⁶⁴ → Linear → 128×7×7 → ConvT → 64×14×14 → ConvT → 1×28×28, Tanh\n"
        f"- **Discriminator** ({counts['discriminator']:,} params): Conv → Conv → Linear → real/fake logit\n"
        "- **Objective:** minimax BCE game with a non-saturating generator loss and one-sided label smoothing\n"
        "- **Optimiser:** Adam, lr 2e-4, β = (0.5, 0.999)\n"
        "- **Data range:** [-1, 1]\n"
        "- **Strengths:** sharp samples. **Risks:** instability, mode collapse, no encoder and no likelihood"
    )
with vae_col, st.container(border=True):
    st.markdown("#### :material/compress: VAE (convolutional)")
    st.markdown(
        f"- **Encoder/decoder** ({counts['vae']:,} params): Conv ×2 → 256 → (μ, log σ²) ∈ ℝ¹⁶ → decoder mirrors the encoder\n"
        "- **Objective:** negative ELBO = Bernoulli reconstruction (BCE) + β·KL(q(z|x) ‖ N(0, I))\n"
        "- **Optimiser:** Adam, lr 1e-3\n"
        "- **Data range:** [0, 1]\n"
        "- **Strengths:** stable training, an encoder, a smooth latent space, a likelihood bound. "
        "**Risks:** blurrier samples"
    )

st.subheader("Architecture")
st.mermaid_chart(ARCHITECTURE_MERMAID)

st.subheader("Dataset")
st.markdown(
    "**MNIST**: 70,000 grayscale 28×28 handwritten digits (60k official train, 10k official test). "
    "The 60k train set is split with a fixed seed into **55,000 training** and **5,000 validation** images. "
    "Generative models train on the training split only. The validation split is the real-data reference "
    "for metrics. The official test set is used once, to report the evaluator classifier's accuracy."
)

st.subheader("Latest experiments")
if summaries.empty:
    empty_state("No experiments yet. Start one from the training page.", "app_pages/train_models.py", "Train models")
else:
    for model in ("gan", "vae"):
        rows = summaries[summaries["model"] == model]
        if rows.empty:
            st.caption(f"No {model.upper()} experiments yet.")
            continue
        latest = rows.iloc[0].to_dict()
        done = rows[rows["state"] == "completed"]
        with st.container(border=True):
            st.markdown(f"**Latest {model.upper()}**: {experiment_label(latest)}  \n{state_text(latest.get('state'))}")
            if done.empty:
                st.caption("No completed run yet, so no metrics to show.")
                continue
            best = done.iloc[0].to_dict()
            st.caption(f"Metrics from the most recent completed run `{best['experiment_id']}`")
            with st.container(horizontal=True):
                st.metric("Validity rate", format_metric(best.get("validity_rate")), border=True)
                st.metric("FD-MNIST (lower is better)", format_metric(best.get("fd_mnist"), 2), border=True)
                st.metric("FID (lower is better)", format_metric(best.get("fid"), 2), border=True)
                st.metric("Training time", format_duration(best.get("train_seconds")), border=True)
