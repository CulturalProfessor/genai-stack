"""Writing the code index into Neo4j: nodes first, then relationships."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from langchain_neo4j import Neo4jGraph

from genai_stack.code_index import (
    batched,
    count_symbols,
    iter_file_rows,
    iter_import_edges,
    iter_reference_edges,
    iter_rows,
)
from genai_stack.schema import (
    CODE_EMBEDDING_PROPERTY,
    CODE_LABEL,
    FILE_LABEL,
    REPOSITORY_LABEL,
)

FILES_QUERY = f"""
UNWIND $rows AS row
MERGE (f:{FILE_LABEL} {{path: row.path}})
    SET f.name = row.name,
        f.language = row.language,
        f.depth = row.depth
"""

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
        c.doc = row.doc,
        c.file_path = row.file_path,
        c.file_name = row.file_name,
        c.text = row.text
WITH f, c, row
CALL db.create.setNodeVectorProperty(c, '{CODE_EMBEDDING_PROPERTY}', row.embedding)
MERGE (f)-[:CONTAINS]->(c)
"""

# Reference edges arrive already resolved to an exact symbol id, by a resolver
# that knows which module each name was imported from and can follow a
# re-export through a barrel. Matching on id rather than on a name and a file
# is what keeps a name declared many times in one file from fanning a single
# reference out across all of them.
#
# One query per relationship type because Cypher cannot parameterise a
# relationship type, and APOC's procedure for it would make this the only place
# in the loader that needs APOC.
def reference_query(relationship: str) -> str:
    return f"""
UNWIND $rows AS row
MATCH (caller:{CODE_LABEL} {{id: row.caller_id}})
MATCH (callee:{CODE_LABEL} {{id: row.callee_id}})
MERGE (caller)-[:{relationship}]->(callee)
"""

# Only an exported symbol can be the target of an import, and knowing which
# ones are exported is what lets a question ask what a module exposes.
EXPORTS_QUERY = f"""
UNWIND $rows AS row
MATCH (f:{FILE_LABEL} {{path: row.file_path}})
MATCH (c:{CODE_LABEL} {{id: row.id}})
MERGE (f)-[:EXPORTS]->(c)
"""

IMPORTS_QUERY = f"""
UNWIND $rows AS row
MATCH (a:{FILE_LABEL} {{path: row.from_path}})
MATCH (b:{FILE_LABEL} {{path: row.to_path}})
MERGE (a)-[:IMPORTS]->(b)
"""


def import_index(
    index: dict,
    graph: Neo4jGraph,
    embeddings,
    batch_size: int = 100,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed every symbol, write it to Neo4j, then link the graph.

    Returns the number of symbols imported.
    """
    total = count_symbols(index)
    imported = 0
    export_batches: list[list[dict]] = []

    # Files first, so a file that defines no symbols still exists for the
    # IMPORTS edges to land on. IMPORT_QUERY only reaches a file through one
    # of its symbols, so on its own it would skip exactly those.
    for rows in batched(iter_file_rows(index), batch_size):
        graph.query(FILES_QUERY, {"rows": rows})

    for batch in batched(iter_rows(index), batch_size):
        # One embedding request per batch instead of one per symbol.
        vectors = embeddings.embed_documents([row["text"] for row in batch])
        for row, vector in zip(batch, vectors):
            row["embedding"] = vector
        graph.query(IMPORT_QUERY, {"rows": batch})
        export_batches.append(
            [
                {"id": row["id"], "file_path": row["file_path"]}
                for row in batch
                if row["exported"]
            ]
        )
        imported += len(batch)
        if on_progress is not None:
            on_progress(imported, total)

    build_relationships(index, export_batches, graph, batch_size)
    return imported


def build_relationships(
    index: dict,
    export_batches: list[list[dict]],
    graph: Neo4jGraph,
    batch_size: int = 100,
) -> None:
    """Create EXPORTS, CALLS and IMPORTS edges once every node exists.

    Relationships are built after the nodes because a call can point at a
    symbol from a file that has not been written yet.
    """
    for rows in export_batches:
        if rows:
            graph.query(EXPORTS_QUERY, {"rows": rows})

    # Grouped by type so each batch is one query. The edges come out of the
    # resolver interleaved, so they are bucketed as they arrive rather than
    # walking the index once per relationship kind.
    buckets: dict[str, list[dict]] = {}
    for edge in iter_reference_edges(index):
        rows = buckets.setdefault(edge["type"], [])
        rows.append({k: v for k, v in edge.items() if k != "type"})
        if len(rows) >= batch_size:
            graph.query(reference_query(edge["type"]), {"rows": rows})
            rows.clear()
    for relationship, rows in buckets.items():
        if rows:
            graph.query(reference_query(relationship), {"rows": rows})

    for rows in batched(iter_import_edges(index), batch_size):
        graph.query(IMPORTS_QUERY, {"rows": rows})


def record_import(graph: Neo4jGraph, index: dict, symbols: int) -> None:
    """Remember what was imported, so the UI can show it without an import."""
    graph.query(
        f"""
        MERGE (r:{REPOSITORY_LABEL} {{name: $name}})
        SET r.source_url = $source_url,
            r.files = $files,
            r.symbols = $symbols,
            r.indexed_at = $indexed_at
        """,
        {
            "name": index.get("repo") or "unknown",
            "source_url": index.get("source_url"),
            "files": len(index.get("files", [])),
            "symbols": symbols,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def list_repositories(graph: Neo4jGraph) -> list[dict]:
    return graph.query(
        f"""
        MATCH (r:{REPOSITORY_LABEL})
        RETURN r.name AS name, r.source_url AS source_url, r.files AS files,
               r.symbols AS symbols, r.indexed_at AS indexed_at
        ORDER BY r.indexed_at DESC
        """
    )


def reset_graph(graph: Neo4jGraph) -> None:
    """Delete every File, Code and Repository node, in batches so large graphs
    don't OOM. Conversations and PDFs are deliberately left alone."""
    while True:
        result = graph.query(
            f"""
            MATCH (n)
            WHERE n:{FILE_LABEL} OR n:{CODE_LABEL} OR n:{REPOSITORY_LABEL}
            WITH n LIMIT 10000
            DETACH DELETE n
            RETURN count(*) AS deleted
            """
        )
        if not result or not result[0]["deleted"]:
            return
