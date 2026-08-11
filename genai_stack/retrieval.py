"""Graph-traversal retrieval over the code knowledge graph.

The vector/keyword search finds an entry point; the Cypher below walks CALLS and
IMPORTS out from that hit to pull in code the question never matched directly.
That traversal is what makes this GraphRAG rather than vector search over a
graph-shaped store - without it the relationships would only be storage.
"""
from __future__ import annotations

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

// What this symbol calls.
OPTIONAL MATCH (node)-[:CALLS]->(callee:{CODE_LABEL})
WITH node, score, f, collect(DISTINCT callee)[..{NEIGHBOUR_LIMIT}] AS callees

// What calls this symbol - the callers are often where the behaviour a
// question asks about actually lives.
OPTIONAL MATCH (caller:{CODE_LABEL})-[:CALLS]->(node)
WITH node, score, f, callees,
     collect(DISTINCT caller)[..{NEIGHBOUR_LIMIT}] AS callers

// Types named in this symbol's signature, and - when the hit is itself a type
// - the code that takes or returns one. Without this a hit on an interface is
// a dead end: it has no calls of its own to walk.
OPTIONAL MATCH (node)-[:REFERENCES_TYPE|EXTENDS|IMPLEMENTS]->(used:{CODE_LABEL})
WITH node, score, f, callees, callers,
     collect(DISTINCT used)[..{NEIGHBOUR_LIMIT}] AS used_types
OPTIONAL MATCH (user:{CODE_LABEL})-[:REFERENCES_TYPE|EXTENDS|IMPLEMENTS]->(node)
WITH node, score, f, callees, callers, used_types,
     collect(DISTINCT user)[..{NEIGHBOUR_LIMIT}] AS type_users

// Files this file depends on.
OPTIONAL MATCH (f)-[:IMPORTS]->(imported:{FILE_LABEL})
WITH node, score, f, callees, callers, used_types, type_users,
     collect(DISTINCT imported.path)[..{NEIGHBOUR_LIMIT}] AS imported_files

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
    {{
        source: coalesce(f.path, node.file_path),
        symbol: node.qualified_name,
        kind: node.kind,
        signature: node.signature,
        lines: toString(node.start_line) + '-' + toString(node.end_line),
        language: f.language,
        calls: [c IN callees | c.qualified_name],
        called_by: [c IN callers | c.qualified_name],
        uses_types: [c IN used_types | c.qualified_name],
        used_as_type_by: [c IN type_users | c.qualified_name],
        imports: imported_files,
        siblings: [(f)-[:CONTAINS]->(other) WHERE other <> node | other.signature][..10]
    }} AS metadata
"""


def build_vector_store(embeddings, url: str, username: str, password: str) -> Neo4jVector:
    """The retriever's entry point: hybrid search over Code nodes.

    Hybrid rather than pure vector because code questions often name a symbol
    exactly, which lexical matching handles better than embeddings.
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
        retrieval_query=CODE_RETRIEVAL_QUERY,
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
