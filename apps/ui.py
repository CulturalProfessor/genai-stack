"""The single UI for the stack.

Replaces three separate Streamlit apps that each ran on their own port and each
loaded their own copy of the embedding model. One app, one port, one model.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="Code graph", page_icon="🕸️", layout="wide")

import page_chat
import page_index
import page_pdf

pages = [
    st.Page(page_index.render, title="Index a repository", icon=":material/hub:"),
    st.Page(page_chat.render, title="Ask the codebase", icon=":material/chat:", default=True),
    st.Page(page_pdf.render, title="Chat with a PDF", icon=":material/description:"),
]

with st.sidebar:
    st.markdown("### Code graph")
    st.caption("Neo4j browser: [localhost:7474](http://localhost:7474)")

st.navigation(pages).run()
