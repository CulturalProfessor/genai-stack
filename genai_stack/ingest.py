"""Writing the code index into Neo4j: nodes first, then relationships."""
from __future__ import annotations

from typing import Callable

from langchain_neo4j import Neo4jGraph

from genai_stack.code_index import (
    batched,
    count_symbols,
    iter_import_edges,
    iter_rows,
)
from genai_stack.schema import (
    CODE_EMBEDDING_PROPERTY,
    CODE_LABEL,
    FILE_LABEL,
)

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
    call_batches: list[list[dict]] = []

    for batch in batched(iter_rows(index), batch_size):
        # One embedding request per batch instead of one per symbol.
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

    build_relationships(index, call_batches, graph, batch_size)
    return imported


def build_relationships(
    index: dict,
    call_batches: list[list[dict]],
    graph: Neo4jGraph,
    batch_size: int = 100,
) -> None:
    """Create CALLS and IMPORTS edges once every node exists."""
    for rows in call_batches:
        if not rows:
            continue
        graph.query(LOCAL_CALLS_QUERY, {"rows": rows})
        graph.query(GLOBAL_CALLS_QUERY, {"rows": rows})

    for rows in batched(iter_import_edges(index), batch_size):
        graph.query(IMPORTS_QUERY, {"rows": rows})


def reset_graph(graph: Neo4jGraph) -> None:
    """Delete every File and Code node, in batches so large graphs don't OOM."""
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
