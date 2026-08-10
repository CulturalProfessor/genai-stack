"""Streamlit UI for importing a code index into the graph."""
import json
import sys
from pathlib import Path

import streamlit as st
from langchain_neo4j import Neo4jGraph
from streamlit.logger import get_logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.code_index import count_symbols, load_index
from genai_stack.ingest import import_index, reset_graph
from genai_stack.llm import load_embedding_model
from genai_stack.schema import apply_schema
from genai_stack.settings import load_settings

logger = get_logger(__name__)
settings = load_settings()


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load the embedding model once per process, not once per rerun."""
    return load_embedding_model(
        settings.embedding_model, config=settings.embedding_config, logger=logger
    )


@st.cache_resource(show_spinner=False)
def get_graph() -> Neo4jGraph:
    """Connect to Neo4j and make sure the schema exists."""
    graph = Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )
    apply_schema(graph, get_embeddings()[1])
    return graph


def render_page() -> None:
    st.header("Code importer")
    st.caption("Go to http://localhost:7474/ to explore the graph.")
    st.markdown(
        f"Generate the index first with `python3 -m genai_stack.extract "
        f"--repo /path/to/repo --output {settings.code_index_path}`."
    )

    uploaded = st.file_uploader("Code index JSON", type="json")
    if uploaded is not None:
        index = json.load(uploaded)
    elif settings.code_index_path.exists():
        index = load_index(settings.code_index_path)
        st.info(f"Using {settings.code_index_path}")
    else:
        st.warning(f"No index found at {settings.code_index_path} - upload one above.")
        return

    st.write(
        f"**{count_symbols(index):,}** symbols across "
        f"**{len(index.get('files', [])):,}** files"
    )

    if st.button("Import", type="primary"):
        progress = st.progress(0.0)
        status = st.empty()

        def on_progress(done: int, total: int) -> None:
            progress.progress(done / total if total else 1.0)
            status.write(f"Embedded and imported {done:,} / {total:,}")

        try:
            with st.spinner("Importing... this can take a few minutes."):
                imported = import_index(
                    index,
                    graph=get_graph(),
                    embeddings=get_embeddings()[0],
                    batch_size=settings.import_batch_size,
                    on_progress=on_progress,
                )
            st.success(f"Imported {imported:,} symbols", icon="✅")
            st.caption("Go to http://localhost:7474/ to interact with the database")
        except Exception as e:
            logger.exception("Import failed")
            st.error(f"Error: {e}", icon="🚨")

    with st.expander("Danger zone"):
        st.caption("Removes every File and Code node so you can re-import cleanly.")
        if st.button("Delete imported graph"):
            try:
                reset_graph(get_graph())
                st.success("Graph cleared", icon="✅")
            except Exception as e:
                logger.exception("Reset failed")
                st.error(f"Error: {e}", icon="🚨")


if __name__ == "__main__":
    render_page()
