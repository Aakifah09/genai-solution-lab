"""GenAI Solution Lab dashboard. Launch with:  streamlit run app.py"""

import streamlit as st

from src.training.jobs import active_job
from src.ui.components import device_info, get_store

st.set_page_config(page_title="GenAI Solution Lab", page_icon=":material/science:", layout="wide")

page = st.navigation(
    {
        "": [
            st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        ],
        "Build": [
            st.Page("app_pages/dataset_explorer.py", title="Dataset explorer", icon=":material/dataset:"),
            st.Page("app_pages/train_models.py", title="Train models", icon=":material/model_training:"),
        ],
        "Results": [
            st.Page("app_pages/generated_samples.py", title="Generated samples", icon=":material/grid_view:"),
            st.Page("app_pages/evaluation.py", title="Evaluation", icon=":material/fact_check:"),
            st.Page("app_pages/comparison.py", title="Experiment comparison", icon=":material/compare_arrows:"),
        ],
        "Docs": [
            st.Page("app_pages/architecture.py", title="Architecture and documentation", icon=":material/account_tree:"),
        ],
    },
    position="sidebar",
)

with st.sidebar:
    info = device_info()
    st.caption("Compute device")
    if info["cuda_available"]:
        st.badge(f"CUDA · {info.get('gpu_name', '')}", icon=":material/memory:", color="green")
    else:
        st.badge("CPU only (CUDA not detected)", icon=":material/memory:", color="orange")
    st.caption(f"torch {info['torch_version']} · Python {info['python_version']}")
    running = active_job(get_store())
    if running:
        st.badge(f"Training: {running}", icon=":material/play_circle:", color="blue")

st.title(page.title, anchor=False)
page.run()
