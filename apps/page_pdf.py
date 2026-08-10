"""Ask questions about an uploaded PDF."""
from __future__ import annotations

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_neo4j import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from streamlit.logger import get_logger

from page_chat import StreamHandler
from resources import get_embeddings, get_llm, get_settings, llm_unavailable

logger = get_logger(__name__)


def render() -> None:
    st.header("Chat with a PDF")
    st.caption(
        "Stored separately from the code graph, under its own `PdfBotChunk` label."
    )

    pdf = st.file_uploader("Upload a PDF", type="pdf")
    if pdf is None:
        return

    settings = get_settings()
    text = "".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
    if not text.strip():
        st.warning("No text could be extracted - is this a scanned PDF?", icon="⚠️")
        return

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    ).split_text(text)
    st.caption(f"{len(chunks)} chunks")

    try:
        store = Neo4jVector.from_texts(
            chunks,
            url=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            embedding=get_embeddings()[0],
            index_name="pdf_bot",
            node_label="PdfBotChunk",
            pre_delete_collection=True,  # Delete existing PDF data
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        logger.exception("Could not embed the PDF")
        st.error(f"Error: {exc}", icon="🚨")
        return

    query = st.text_input("Ask a question about this PDF")
    if not query:
        return

    try:
        qa = RetrievalQA.from_chain_type(
            llm=get_llm(), chain_type="stuff", retriever=store.as_retriever()
        )
        qa.run(query, callbacks=[StreamHandler(st.empty())])
    except Exception as exc:  # noqa: BLE001 - the page explains and stays up
        logger.exception("PDF query failed")
        llm_unavailable(exc)
