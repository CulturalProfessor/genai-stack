"""Sidebar chrome every page shares: what it is, and what is reachable.

Rendered after the page so its own sections (conversations, PDFs) come first
and the status stays pinned at the bottom of the sidebar.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import streamlit as st

from resources import get_settings

# Names load_llm treats as hosted providers; anything else is an Ollama tag.
HOSTED = {"gpt-4": "OpenAI", "gpt-3.5": "OpenAI", "claude": "AWS Bedrock",
          "claudev2": "AWS Bedrock"}


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models(base_url: str) -> list[str] | None:
    """Model tags the server has, or None if it cannot be reached.

    Cached briefly: this runs on every rerun, and a chat page reruns on every
    keystroke of a rename.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return [model.get("name", "") for model in payload.get("models", [])]


def llm_status(settings) -> tuple[str, str]:
    """(icon, message) describing whether questions can be answered."""
    model = settings.llm or ""
    if model in HOSTED:
        # No live check: a request would cost money and a key can be valid
        # without this page knowing it.
        return "☁️", f"{HOSTED[model]} · `{model}`"

    if not settings.ollama_base_url:
        return "⚠️", "No OLLAMA_BASE_URL set"

    available = ollama_models(settings.ollama_base_url)
    if available is None:
        return "🔴", f"Ollama unreachable at {settings.ollama_base_url}"
    if model not in available:
        return "🟠", f"`{model}` not pulled on the Ollama server"
    return "🟢", f"Ollama · `{model}`"


def neo4j_status() -> tuple[str, str]:
    from resources import get_graph

    try:
        get_graph().query("RETURN 1")
    except Exception as exc:  # noqa: BLE001 - the status line is the report
        return "🔴", f"Neo4j unreachable: {type(exc).__name__}"
    return "🟢", "Neo4j connected"


def sidebar_status() -> None:
    settings = get_settings()
    with st.sidebar:
        st.divider()
        icon, message = neo4j_status()
        st.caption(f"{icon} {message}")
        icon, message = llm_status(settings)
        st.caption(f"{icon} {message}")
        st.caption("Browse the graph at [localhost:7474](http://localhost:7474)")
