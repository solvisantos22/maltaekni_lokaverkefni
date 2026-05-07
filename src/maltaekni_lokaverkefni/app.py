"""Streamlit app for the Icelandic consumer-rights RAG MVP."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    from .answer_generator import generate_grounded_answer
    from .retriever import build_retriever
except ImportError:  # Allows direct script execution during early experiments.
    from answer_generator import generate_grounded_answer
    from retriever import build_retriever


CHUNKS_PATH = Path("data/processed/chunks.json")


@st.cache_resource
def _load_retriever(method: str):
    return build_retriever(method, chunks_path=CHUNKS_PATH)


def main():
    st.set_page_config(page_title="Neytendaréttur", page_icon="§", layout="wide")
    st.title("Neytendaréttur")

    if not CHUNKS_PATH.exists():
        st.error("Vantar data/processed/chunks.json.")
        st.code(
            "python src\\maltaekni_lokaverkefni\\fetch_sources.py\n"
            "python src\\maltaekni_lokaverkefni\\chunking.py",
            language="powershell",
        )
        return

    with st.sidebar:
        method = st.selectbox("Leitaraðferð", ["tfidf", "bm25"], index=0)
        top_k = st.slider("Fjöldi heimilda", min_value=1, max_value=5, value=3)
        show_prompt = st.toggle("Sýna prompt", value=False)

    question = st.text_area(
        "Spurning",
        value="Hvað get ég gert ef vara sem ég keypti er gölluð?",
        height=110,
    )

    if st.button("Svara", type="primary") and question.strip():
        retriever = _load_retriever(method)
        retrieval_result = retriever.search(question.strip(), top_k=top_k)
        answer_result = generate_grounded_answer(retrieval_result, max_sources=top_k)

        st.subheader("Svar")
        st.write(answer_result.answer)
        st.caption(f"Traust: {answer_result.confidence} | Leit: {method}")

        st.subheader("Heimildir")
        for source in answer_result.sources:
            label = f"[{source.citation_id}] {source.title} - {source.section}"
            with st.expander(label, expanded=source.citation_id == 1):
                st.write(source.text)
                st.write(source.url)
                if source.score is not None:
                    st.caption(f"Score: {source.score:.4f}")

        if show_prompt:
            st.subheader("Prompt")
            st.text_area("System", value=answer_result.system_prompt, height=160)
            st.text_area("User", value=answer_result.user_prompt, height=360)


if __name__ == "__main__":
    main()
