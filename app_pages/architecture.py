import re

import streamlit as st

from src.ui.helpers import ARCHITECTURE_MERMAID
from src.utils.config import PROJECT_ROOT

st.mermaid_chart(ARCHITECTURE_MERMAID)

docs = {
    "Architecture": PROJECT_ROOT / "docs" / "architecture.md",
    "Methodology": PROJECT_ROOT / "docs" / "methodology.md",
    "Evaluation": PROJECT_ROOT / "docs" / "evaluation.md",
}
choice = st.segmented_control("Document", list(docs), default="Architecture", key="doc_choice", required=True)
path = docs[choice]
if not path.exists():
    st.error(f"Missing documentation file: {path}")
else:
    text = path.read_text(encoding="utf-8")
    # The diagram is already drawn above; drop the duplicate mermaid block and the H1 title.
    text = re.sub(r"```mermaid.*?```", "", text, count=1, flags=re.S)
    text = re.sub(r"^# .*\n", "", text, count=1)
    st.markdown(text)
