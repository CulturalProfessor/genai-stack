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
from chrome import sidebar_status

# Every page module names its entrypoint `render`, so Streamlit would infer the
# same URL path for all three. Spell the paths out instead.
pages = [
    st.Page(page_index.render, title="Index a repository", icon=":material/hub:", url_path="index"),
    st.Page(page_chat.render, title="Ask the codebase", icon=":material/chat:", url_path="chat", default=True),
    st.Page(page_pdf.render, title="Chat with a PDF", icon=":material/description:", url_path="pdf"),
]

st.navigation(pages).run()

# After the page, so its own sidebar sections come first and the status sits
# at the bottom of the sidebar rather than above them.
sidebar_status()
