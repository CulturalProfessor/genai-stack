"""Graph-traversal retrieval over the code knowledge graph.

The vector/keyword search finds an entry point; the Cypher below walks CALLS,
the type edges and IMPORTS out from that hit to pull in code the question never
matched directly.
That traversal is what makes this GraphRAG rather than vector search over a
graph-shaped store - without it the relationships would only be storage.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_neo4j import Neo4jVector

from genai_stack.pdf_store import PDF_CHUNK_LABEL, PDF_INDEX_NAME
from genai_stack.schema import (
    CODE_EMBEDDING_PROPERTY,
    CODE_LABEL,
    CODE_TEXT_PROPERTY,
    FILE_LABEL,
    FULLTEXT_INDEX_NAME,
    VECTOR_INDEX_NAME,
)

# How many neighbours to fold into one hit's context. Small on purpose: the
# point is to name where related logic lives, not to inline half the repo.
NEIGHBOUR_LIMIT = 5

# `node` and `score` are bound by Neo4jVector from the vector/keyword search.
# The query must return `text`, `score` and `metadata`; `metadata.source` is
# what the chain cites.
CODE_RETRIEVAL_QUERY = f"""
WITH node, score

// The file the matched symbol lives in.
OPTIONAL MATCH (f:{FILE_LABEL})-[:CONTAINS]->(node)

// Each neighbour set is gathered in its own subquery, and every ORDER BY below
// lives inside one. A bare ORDER BY at this level re-sorts the entire result
// stream: the degree ordering added to pick better neighbours also threw away
// the relevance ranking the hybrid search had just established, and every
// document came back sorted by how connected its callers were. The eval caught
// it as the expansion arm scoring below the no-expansion arm on recall, which
// is impossible when both run the same search.

// What this symbol calls, least connected first. The cap used to take an
// arbitrary five in whatever order the planner returned, so a generic helper
// crowded out the specific collaborator that answers the question.
CALL (node) {{
    OPTIONAL MATCH (node)-[:CALLS]->(callee:{CODE_LABEL})
    WITH callee, CASE WHEN callee IS NULL THEN 0
                      ELSE count {{ (callee)<-[:CALLS]-() }} END AS degree
    ORDER BY degree ASC
    RETURN collect(DISTINCT callee)[..{NEIGHBOUR_LIMIT}] AS callees
}}

// What calls this symbol - the callers are often where the behaviour a
// question asks about actually lives.
CALL (node) {{
    OPTIONAL MATCH (caller:{CODE_LABEL})-[:CALLS]->(node)
    WITH caller, CASE WHEN caller IS NULL THEN 0
                      ELSE count {{ (caller)-[:CALLS]->() }} END AS degree
    ORDER BY degree ASC
    RETURN collect(DISTINCT caller)[..{NEIGHBOUR_LIMIT}] AS callers
}}

// Types named in this symbol's signature, and - when the hit is itself a type
// - the code that takes or returns one. Without this a hit on an interface is
// a dead end: it has no calls of its own to walk.
CALL (node) {{
    OPTIONAL MATCH (node)-[:REFERENCES_TYPE|EXTENDS|IMPLEMENTS]->(used:{CODE_LABEL})
    RETURN collect(DISTINCT used)[..{NEIGHBOUR_LIMIT}] AS used_types
}}
CALL (node) {{
    OPTIONAL MATCH (user:{CODE_LABEL})-[:REFERENCES_TYPE|EXTENDS|IMPLEMENTS]->(node)
    RETURN collect(DISTINCT user)[..{NEIGHBOUR_LIMIT}] AS type_users
}}

// Files this file depends on.
CALL (f) {{
    OPTIONAL MATCH (f)-[:IMPORTS]->(imported:{FILE_LABEL})
    RETURN collect(DISTINCT imported.path)[..{NEIGHBOUR_LIMIT}] AS imported_files
}}

WITH node, score, f, callees, callers, used_types, type_users, imported_files

WITH node, score, f, callees, callers, used_types, type_users, imported_files,
     [c IN callees |
        '  ' + c.qualified_name + '  (' + c.file_path + ':' +
        toString(c.start_line) + ')  ' + coalesce(c.signature, '')] AS callee_lines,
     [c IN callers |
        '  ' + c.qualified_name + '  (' + c.file_path + ':' +
        toString(c.start_line) + ')  ' + coalesce(c.signature, '')] AS caller_lines,
     [c IN used_types |
        '  ' + c.qualified_name + '  (' + c.file_path + ':' +
        toString(c.start_line) + ')'] AS used_type_lines,
     [c IN type_users |
        '  ' + c.qualified_name + '  (' + c.file_path + ':' +
        toString(c.start_line) + ')  ' + coalesce(c.signature, '')] AS type_user_lines

RETURN
    node.{CODE_TEXT_PROPERTY}
    + CASE WHEN size(callee_lines) > 0
        THEN '\\n\\n// Calls:\\n' + apoc.text.join(callee_lines, '\\n')
        ELSE '' END
    + CASE WHEN size(caller_lines) > 0
        THEN '\\n\\n// Called by:\\n' + apoc.text.join(caller_lines, '\\n')
        ELSE '' END
    + CASE WHEN size(used_type_lines) > 0
        THEN '\\n\\n// Types used:\\n' + apoc.text.join(used_type_lines, '\\n')
        ELSE '' END
    + CASE WHEN size(type_user_lines) > 0
        THEN '\\n\\n// Used as a type by:\\n' + apoc.text.join(type_user_lines, '\\n')
        ELSE '' END
    + CASE WHEN size(imported_files) > 0
        THEN '\\n\\n// This file imports: ' + apoc.text.join(imported_files, ', ')
        ELSE '' END
    AS text,
    score,
    // Neighbours carry the file they live in as well as their name. A
    // qualified name alone does not identify a symbol - `generator` occurs in
    // many files - so scoring a context against gold needs the pair.
    {{
        source: coalesce(f.path, node.file_path),
        symbol: node.qualified_name,
        kind: node.kind,
        signature: node.signature,
        lines: toString(node.start_line) + '-' + toString(node.end_line),
        language: f.language,
        calls: [c IN callees | {{file: c.file_path, symbol: c.qualified_name}}],
        called_by: [c IN callers | {{file: c.file_path, symbol: c.qualified_name}}],
        uses_types: [c IN used_types | {{file: c.file_path, symbol: c.qualified_name}}],
        used_as_type_by: [c IN type_users | {{file: c.file_path, symbol: c.qualified_name}}],
        imports: imported_files,
        siblings: [(f)-[:CONTAINS]->(other) WHERE other <> node | other.signature][..10]
    }} AS metadata
"""

# The same rows without walking a single edge. Subtracting what this retrieves
# from what the query above retrieves is the graph lift: the measurement that
# says whether the traversal earns its latency, or whether this is vector
# search that happens to store its chunks in a graph. It is only ever used by
# the eval - nothing in the app should retrieve without expanding.
CODE_SEEDS_ONLY_QUERY = f"""
WITH node, score
OPTIONAL MATCH (f:{FILE_LABEL})-[:CONTAINS]->(node)
RETURN
    node.{CODE_TEXT_PROPERTY} AS text,
    score,
    {{
        source: coalesce(f.path, node.file_path),
        symbol: node.qualified_name,
        kind: node.kind,
        signature: node.signature,
        lines: toString(node.start_line) + '-' + toString(node.end_line),
        language: f.language
    }} AS metadata
"""


def build_vector_store(
    embeddings, url: str, username: str, password: str, *, expand: bool = True
) -> Neo4jVector:
    """The retriever's entry point: hybrid search over Code nodes.

    Hybrid rather than pure vector because code questions often name a symbol
    exactly, which lexical matching handles better than embeddings.

    `expand=False` swaps in the seeds-only query. Same search, same ranking,
    no traversal - the control arm the eval measures graph lift against.
    """
    return Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=url,
        username=username,
        password=password,
        database="neo4j",  # neo4j by default
        index_name=VECTOR_INDEX_NAME,
        keyword_index_name=FULLTEXT_INDEX_NAME,
        search_type="hybrid",
        node_label=CODE_LABEL,
        text_node_property=CODE_TEXT_PROPERTY,
        embedding_node_property=CODE_EMBEDDING_PROPERTY,
        retrieval_query=CODE_RETRIEVAL_QUERY if expand else CODE_SEEDS_ONLY_QUERY,
    )


def build_document_store(embeddings, url: str, username: str, password: str) -> Neo4jVector:
    """Plain vector search over the chunks of uploaded PDFs.

    No traversal and no keyword index: prose has no call graph to walk, and a
    chunk is a slice of text rather than a named symbol to match exactly.
    The caller filters this down to the documents attached to one chat.
    """
    return Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=url,
        username=username,
        password=password,
        index_name=PDF_INDEX_NAME,
        node_label=PDF_CHUNK_LABEL,
    )


# One store per expansion mode, built on first use. `from_existing_index`
# round-trips to Neo4j to check the index dimension, so an eval asking fifty
# questions should not pay for that fifty times.
_stores: dict[bool, Neo4jVector] = {}


def _default_store(expand: bool) -> Neo4jVector:
    from genai_stack.llm import load_embedding_model
    from genai_stack.settings import load_settings

    if expand not in _stores:
        settings = load_settings()
        embeddings, _ = load_embedding_model(
            settings.embedding_model, config=settings.embedding_config
        )
        _stores[expand] = build_vector_store(
            embeddings,
            settings.neo4j_uri,
            settings.neo4j_username,
            settings.neo4j_password,
            expand=expand,
        )
    return _stores[expand]


def retrieve(
    question: str,
    k: int = 4,
    expand: bool = True,
    *,
    store: Neo4jVector | None = None,
) -> list[Document]:
    """Retrieve the context for one question, with no chain around it.

    The app reaches retrieval through `ChatRetrievalQAWithSourcesChain`, which
    widens the question with the conversation, calls an LLM and returns prose.
    None of that is measurable. This is the same retrieval with the two ends
    cut off, so the eval can score what actually lands in the context.

    The similarity score is copied onto each document's metadata, since the
    gap between the top score on an answerable question and on one with no
    answer in the repo is the only abstention signal available before the LLM.
    """
    store = store if store is not None else _default_store(expand)
    documents = []
    for document, score in store.similarity_search_with_score(question, k=k):
        document.metadata["score"] = score
        documents.append(document)
    return documents
