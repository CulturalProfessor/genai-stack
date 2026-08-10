"""Index a repository into the code graph."""
from __future__ import annotations

import json

import streamlit as st
from streamlit.logger import get_logger

from genai_stack.code_index import count_symbols, load_index
from genai_stack.extract import build_payload
from genai_stack.ingest import import_index, reset_graph
from genai_stack.sources import allowed_hosts, cloned_repository, parse_repo_url
from resources import get_embeddings, get_graph, get_settings

logger = get_logger(__name__)


def describe(index: dict) -> str:
    return (
        f"**{count_symbols(index):,}** symbols across "
        f"**{len(index.get('files', [])):,}** files"
    )


def import_to_graph(index: dict) -> int:
    progress = st.progress(0.0)
    status = st.empty()

    def on_progress(done: int, total: int) -> None:
        progress.progress(done / total if total else 1.0)
        status.write(f"Embedded and imported {done:,} / {total:,} symbols")

    imported = import_index(
        index,
        graph=get_graph(),
        embeddings=get_embeddings()[0],
        batch_size=get_settings().import_batch_size,
        on_progress=on_progress,
    )
    progress.empty()
    status.empty()
    return imported


def graph_summary() -> None:
    """What landed in the graph. Cross-file call edges are the number to watch."""
    try:
        counts = get_graph().query(
            """
            MATCH (f:File) WITH count(f) AS files
            MATCH (c:Code) WITH files, count(c) AS symbols
            OPTIONAL MATCH (a:Code)-[r:CALLS]->(b:Code)
            WITH files, symbols, count(r) AS calls,
                 count(CASE WHEN a.file_path <> b.file_path THEN 1 END) AS cross_file
            OPTIONAL MATCH (:File)-[i:IMPORTS]->(:File)
            RETURN files, symbols, calls, cross_file, count(i) AS imports
            """
        )[0]
    except Exception:  # noqa: BLE001 - a summary is a nicety, not the result
        logger.exception("Could not read graph summary")
        return

    a, b, c, d = st.columns(4)
    a.metric("Files", f"{counts['files']:,}")
    b.metric("Symbols", f"{counts['symbols']:,}")
    c.metric(
        "Call edges", f"{counts['calls']:,}", f"{counts['cross_file']:,} cross-file"
    )
    d.metric("Import edges", f"{counts['imports']:,}")
    st.caption(
        "Cross-file call edges are what retrieval traverses to reach code a "
        "question does not name directly."
    )


def from_url() -> None:
    settings = get_settings()
    st.caption(
        "Public repositories only. The code is parsed, never executed - no "
        "install or build step runs."
    )
    url = st.text_input(
        "Repository URL",
        placeholder="https://github.com/owner/repo",
        help=f"Also accepts owner/repo. Allowed hosts: {', '.join(allowed_hosts())}",
    )
    col1, col2 = st.columns([2, 3])
    branch = col1.text_input("Branch", placeholder="default branch")
    keep = col2.checkbox(
        "Save the index to disk",
        value=True,
        help=f"Writes {settings.code_index_path} so you can re-import without cloning again.",
    )

    if not st.button("Index repository", type="primary", disabled=not url):
        return

    try:
        ref = parse_repo_url(url, branch or None)
    except ValueError as exc:
        st.error(str(exc), icon="🚨")
        return

    try:
        with st.status(f"Indexing {ref.slug}", expanded=True) as status:
            st.write("Cloning (shallow, single branch)...")
            with cloned_repository(ref) as checkout:
                st.write("Parsing with tree-sitter...")
                index = build_payload(
                    checkout, repo_name=ref.name, source_url=ref.clone_url
                )

            if not count_symbols(index):
                status.update(label=f"No code found in {ref.slug}", state="error")
                st.warning(
                    "No TypeScript or JavaScript symbols were found. This importer "
                    "only understands .ts, .tsx, .js and .jsx files.",
                    icon="⚠️",
                )
                return
            st.write(f"Found {describe(index)}")

            if keep:
                settings.code_index_path.parent.mkdir(parents=True, exist_ok=True)
                settings.code_index_path.write_text(
                    json.dumps(index, indent=2), encoding="utf-8"
                )
                st.write(f"Saved {settings.code_index_path}")

            st.write("Embedding symbols and building the graph...")
            imported = import_to_graph(index)
            status.update(label=f"Indexed {ref.slug}", state="complete")

        st.success(f"Imported {imported:,} symbols from {ref.slug}", icon="✅")
        graph_summary()
    except RuntimeError as exc:  # clone failures carry a readable message
        st.error(str(exc), icon="🚨")
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        logger.exception("Indexing failed")
        st.error(f"Error: {exc}", icon="🚨")


def from_file() -> None:
    settings = get_settings()
    uploaded = st.file_uploader("Code index JSON", type="json")
    if uploaded is not None:
        index = json.load(uploaded)
    elif settings.code_index_path.exists():
        index = load_index(settings.code_index_path)
        st.info(f"Using {settings.code_index_path}")
    else:
        st.caption(
            f"No index at {settings.code_index_path} yet. Index a repository from "
            "a URL, or generate one with "
            "`python3 -m genai_stack.extract --repo /path/to/repo`."
        )
        return

    st.write(describe(index))
    if st.button("Import", type="primary"):
        try:
            imported = import_to_graph(index)
            st.success(f"Imported {imported:,} symbols", icon="✅")
            graph_summary()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            logger.exception("Import failed")
            st.error(f"Error: {exc}", icon="🚨")


def render() -> None:
    st.header("Index a repository")
    st.caption("Explore the result at http://localhost:7474")

    url_tab, file_tab = st.tabs(["From a repository URL", "From an index file"])
    with url_tab:
        from_url()
    with file_tab:
        from_file()

    with st.expander("Danger zone"):
        st.caption("Removes every File and Code node so you can re-import cleanly.")
        if st.button("Delete imported graph"):
            try:
                reset_graph(get_graph())
                st.success("Graph cleared", icon="✅")
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                logger.exception("Reset failed")
                st.error(f"Error: {exc}", icon="🚨")
