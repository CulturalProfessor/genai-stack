"""Shared, cached resources for the UI.

Everything here is `@st.cache_resource`, so the embedding model, the database
connection and the chains are created once for the whole app rather than once
per page. The three Streamlit apps this replaced each loaded their own copy of
the embedding model into their own container.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from langchain_neo4j import Neo4jGraph
from streamlit.logger import get_logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.chains import configure_llm_only_chain, configure_qa_rag_chain
from genai_stack.llm import load_embedding_model, load_llm
from genai_stack.schema import apply_schema
from genai_stack.settings import Settings, load_settings

logger = get_logger(__name__)


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return load_settings()


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embeddings():
    s = get_settings()
    return load_embedding_model(s.embedding_model, config=s.embedding_config, logger=logger)


@st.cache_resource(show_spinner="Connecting to Neo4j...")
def get_graph() -> Neo4jGraph:
    s = get_settings()
    graph = Neo4jGraph(
        url=s.neo4j_uri, username=s.neo4j_username, password=s.neo4j_password
    )
    apply_schema(graph, get_embeddings()[1])
    return graph


@st.cache_resource(show_spinner=False)
def get_llm():
    s = get_settings()
    return load_llm(s.llm, logger=logger, config=s.embedding_config)


@st.cache_resource(show_spinner=False)
def get_chains():
    """(llm_only, rag). Only the pages that answer questions need these."""
    s = get_settings()
    llm = get_llm()
    return (
        configure_llm_only_chain(llm),
        configure_qa_rag_chain(
            llm,
            get_embeddings()[0],
            embeddings_store_url=s.neo4j_uri,
            username=s.neo4j_username,
            password=s.neo4j_password,
        ),
    )


def llm_unavailable(exc: Exception) -> None:
    """Explain a missing LLM without taking the rest of the app down with it."""
    st.error(f"The language model is not reachable: {exc}", icon="🚨")
    st.info(
        "Indexing a repository needs no LLM and still works. To answer questions, "
        "either run Ollama with `docker compose --profile linux up` and set "
        "`OLLAMA_BASE_URL=http://llm:11434`, or set `LLM=gpt-3.5` and "
        "`OPENAI_API_KEY` in `.env`.",
        icon="💡",
    )
