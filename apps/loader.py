"""Import extracted code symbols into Neo4j, with embeddings for retrieval.

Reads the JSON produced by ``fs.py`` and writes a ``(:File)-[:CONTAINS]->(:Code)``
graph where every Code node carries the embedding of its own source text.
Run it through the Streamlit UI (``docker compose up loader``) or import
``import_index`` from another script.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import streamlit as st
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from streamlit.logger import get_logger

from chains import load_embedding_model
from code_index import (
    batched,
    count_symbols,
    iter_import_edges,
    iter_rows,
    load_index,
)
from utils import (
    CODE_EMBEDDING_PROPERTY,
    CODE_LABEL,
    FILE_LABEL,
    create_constraints,
    create_fulltext_index,
    create_vector_index,
)

load_dotenv(".env")

logger = get_logger(__name__)

CODE_INDEX_PATH = Path(os.getenv("CODE_INDEX_PATH", "data/all_code_blocks.json"))
BATCH_SIZE = int(os.getenv("IMPORT_BATCH_SIZE", "100"))

IMPORT_QUERY = f"""
UNWIND $rows AS row
MERGE (f:{FILE_LABEL} {{path: row.file_path}})
    SET f.name = row.file_name,
        f.language = row.language,
        f.depth = row.depth
MERGE (c:{CODE_LABEL} {{id: row.id}})
    SET c.name = row.name,
        c.qualified_name = row.qualified_name,
        c.kind = row.kind,
        c.parent = row.parent,
        c.start_line = row.start_line,
        c.end_line = row.end_line,
        c.signature = row.signature,
        c.file_path = row.file_path,
        c.file_name = row.file_name,
        c.text = row.text
WITH f, c, row
CALL db.create.setNodeVectorProperty(c, '{CODE_EMBEDDING_PROPERTY}', row.embedding)
MERGE (f)-[:CONTAINS]->(c)
"""

# Call edges are resolved by name in two passes. A definition in the same file
# always wins; otherwise the name must be unique across the repo, because
# linking every `handle` in the codebase to every caller of `handle` would bury
# the real relationships in noise.
LOCAL_CALLS_QUERY = f"""
UNWIND $rows AS row
MATCH (caller:{CODE_LABEL} {{id: row.id}})
UNWIND row.calls AS callee_name
MATCH (callee:{CODE_LABEL} {{name: callee_name, file_path: row.file_path}})
WHERE callee <> caller
MERGE (caller)-[:CALLS]->(callee)
"""

GLOBAL_CALLS_QUERY = f"""
UNWIND $rows AS row
MATCH (caller:{CODE_LABEL} {{id: row.id}})
UNWIND row.calls AS callee_name
WITH caller, row, callee_name
WHERE NOT EXISTS {{
    MATCH (:{CODE_LABEL} {{name: callee_name, file_path: row.file_path}})
}}
CALL (callee_name) {{
    MATCH (candidate:{CODE_LABEL} {{name: callee_name}})
    WITH collect(candidate) AS candidates
    RETURN CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS callee
}}
WITH caller, callee
WHERE callee IS NOT NULL AND callee <> caller
MERGE (caller)-[:CALLS]->(callee)
"""

IMPORTS_QUERY = f"""
UNWIND $rows AS row
MATCH (a:{FILE_LABEL} {{path: row.from_path}})
MATCH (b:{FILE_LABEL} {{path: row.to_path}})
MERGE (a)-[:IMPORTS]->(b)
"""


@st.cache_resource(show_spinner=False)
def get_graph() -> Neo4jGraph:
    """Connect to Neo4j and make sure the schema exists.

    Cached so Streamlit reruns reuse one connection.
    """
    url = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not url:
        raise RuntimeError("NEO4J_URI is not set - copy env.example to .env first")
    # Remapping for Langchain Neo4j integration
    os.environ["NEO4J_URL"] = url

    graph = Neo4jGraph(url=url, username=username, password=password)
    _, dimension = get_embeddings()
    create_constraints(graph)
    create_vector_index(graph, dimension)
    create_fulltext_index(graph)
    return graph


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load the embedding model once per process, not once per rerun."""
    return load_embedding_model(
        os.getenv("EMBEDDING_MODEL"),
        config={"ollama_base_url": os.getenv("OLLAMA_BASE_URL")},
        logger=logger,
    )


def import_index(
    index: dict,
    graph: Neo4jGraph | None = None,
    embeddings=None,
    batch_size: int = BATCH_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed every symbol and write it to Neo4j. Returns the number imported."""
    graph = graph if graph is not None else get_graph()
    embeddings = embeddings if embeddings is not None else get_embeddings()[0]

    total = count_symbols(index)
    imported = 0
    call_batches: list[list[dict]] = []
    for batch in batched(iter_rows(index), batch_size):
        # One request per batch instead of one per symbol.
        vectors = embeddings.embed_documents([row["text"] for row in batch])
        for row, vector in zip(batch, vectors):
            row["embedding"] = vector
        graph.query(IMPORT_QUERY, {"rows": batch})
        # Relationships are built afterwards: a call can point at a symbol from
        # a file that has not been written yet.
        call_batches.append(
            [
                {"id": row["id"], "file_path": row["file_path"], "calls": row["calls"]}
                for row in batch
                if row["calls"]
            ]
        )
        imported += len(batch)
        if on_progress is not None:
            on_progress(imported, total)

    build_relationships(index, call_batches, graph)
    return imported


def build_relationships(
    index: dict, call_batches: list[list[dict]], graph: Neo4jGraph
) -> None:
    """Create CALLS and IMPORTS edges once every node exists."""
    for rows in call_batches:
        if not rows:
            continue
        graph.query(LOCAL_CALLS_QUERY, {"rows": rows})
        graph.query(GLOBAL_CALLS_QUERY, {"rows": rows})

    for rows in batched(iter_import_edges(index), BATCH_SIZE):
        graph.query(IMPORTS_QUERY, {"rows": rows})


def reset_graph(graph: Neo4jGraph | None = None) -> None:
    """Delete every File and Code node, in batches so large graphs don't OOM."""
    graph = graph if graph is not None else get_graph()
    while True:
        result = graph.query(
            f"""
            MATCH (n)
            WHERE n:{FILE_LABEL} OR n:{CODE_LABEL}
            WITH n LIMIT 10000
            DETACH DELETE n
            RETURN count(*) AS deleted
            """
        )
        if not result or not result[0]["deleted"]:
            return


def render_page() -> None:
    st.header("Code importer")
    st.caption("Go to http://localhost:7474/ to explore the graph.")

    st.markdown(
        f"Generate the index first with "
        f"`python3 fs.py --repo /path/to/repo --output {CODE_INDEX_PATH}`."
    )

    uploaded = st.file_uploader("Code index JSON", type="json")
    index = None
    if uploaded is not None:
        index = json.load(uploaded)
    elif CODE_INDEX_PATH.exists():
        index = load_index(CODE_INDEX_PATH)
        st.info(f"Using {CODE_INDEX_PATH}")
    else:
        st.warning(f"No index found at {CODE_INDEX_PATH} - upload one above.")

    if index is None:
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
                imported = import_index(index, on_progress=on_progress)
            st.success(f"Imported {imported:,} symbols", icon="✅")
            st.caption("Go to http://localhost:7474/ to interact with the database")
        except Exception as e:
            logger.exception("Import failed")
            st.error(f"Error: {e}", icon="🚨")

    with st.expander("Danger zone"):
        st.caption("Removes every File and Code node so you can re-import cleanly.")
        if st.button("Delete imported graph"):
            try:
                reset_graph()
                st.success("Graph cleared", icon="✅")
            except Exception as e:
                logger.exception("Reset failed")
                st.error(f"Error: {e}", icon="🚨")


if __name__ == "__main__":
    render_page()
