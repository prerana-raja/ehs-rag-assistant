"""Streamlit UI: streamlit run app.py"""
import streamlit as st
from src.ingest import load_chunks, DOCS_DIR
from src.retriever import Retriever
from src.generate import answer

st.set_page_config(page_title="EHS Assistant", page_icon="🦺")
st.title("🦺 EHS Procedure Assistant")
st.caption("Ask questions about the safety documents in data/docs. Sample content is fictional.")


@st.cache_resource
def get_retriever():
    return Retriever(load_chunks(DOCS_DIR))


try:
    retriever = get_retriever()
except (ValueError, FileNotFoundError) as e:
    st.error(str(e))
    st.stop()

q = st.text_input("Your question", placeholder="When is a hot work permit required?")
if q:
    hits = retriever.search(q)
    with st.spinner("Thinking..."):
        st.markdown(answer(q, hits))
    with st.expander("Retrieved sources"):
        for i, (c, s) in enumerate(hits, 1):
            st.markdown(f"**[{i}] {c.label}** (score {s:.2f})")
            st.write(c.text)
