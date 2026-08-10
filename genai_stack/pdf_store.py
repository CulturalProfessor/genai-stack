"""Uploaded PDFs, kept side by side instead of one at a time.

The chunks live in the same `pdf_bot` vector index they always did, but each
one now carries the id of the document it came from, so a question can be
answered from one PDF without deleting the others first.

    (:PdfDocument {id, name, chunks, pages, uploaded_at})
    (:PdfBotChunk {text, embedding, document_id, document_name})

There is no relationship between the two: the chunks are written by
Neo4jVector, which only knows how to set properties from a document's
metadata. `document_id` is that link, and what retrieval filters on.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

PDF_DOCUMENT_LABEL = "PdfDocument"
PDF_CHUNK_LABEL = "PdfBotChunk"
PDF_INDEX_NAME = "pdf_bot"
DOCUMENT_ID_PROPERTY = "document_id"


@dataclass(frozen=True)
class PdfDocument:
    id: str
    name: str
    chunks: int
    pages: int
    uploaded_at: str


def document_id(name: str, content: bytes) -> str:
    """Content-addressed, so re-uploading a file finds its old conversations.

    The name is part of it: the same paper saved under two names is two
    documents to a reader picking one from a list.
    """
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(content)
    return digest.hexdigest()[:32]


def create_constraints(graph) -> None:
    graph.query(
        f"CREATE CONSTRAINT pdf_document_id IF NOT EXISTS "
        f"FOR (d:{PDF_DOCUMENT_LABEL}) REQUIRE d.id IS UNIQUE"
    )


def register_document(
    graph, document_id: str, name: str, chunks: int, pages: int
) -> None:
    """Record an uploaded PDF, or refresh it if the same file comes back."""
    graph.query(
        f"""
        MERGE (d:{PDF_DOCUMENT_LABEL} {{id: $id}})
        SET d.name = $name, d.chunks = $chunks, d.pages = $pages,
            d.uploaded_at = $uploaded_at
        """,
        {
            "id": document_id,
            "name": name,
            "chunks": chunks,
            "pages": pages,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def list_documents(graph) -> list[PdfDocument]:
    rows = graph.query(
        f"""
        MATCH (d:{PDF_DOCUMENT_LABEL})
        RETURN d.id AS id, d.name AS name, d.chunks AS chunks,
               d.pages AS pages, d.uploaded_at AS uploaded_at
        ORDER BY d.uploaded_at DESC
        """
    )
    return [PdfDocument(**row) for row in rows]


def get_document(graph, document_id: str) -> PdfDocument | None:
    rows = graph.query(
        f"""
        MATCH (d:{PDF_DOCUMENT_LABEL} {{id: $id}})
        RETURN d.id AS id, d.name AS name, d.chunks AS chunks,
               d.pages AS pages, d.uploaded_at AS uploaded_at
        """,
        {"id": document_id},
    )
    return PdfDocument(**rows[0]) if rows else None


def delete_document(graph, document_id: str) -> None:
    """Remove a PDF and its chunks. Conversations about it go separately."""
    graph.query(
        f"""
        MATCH (c:{PDF_CHUNK_LABEL} {{{DOCUMENT_ID_PROPERTY}: $id}})
        DETACH DELETE c
        """,
        {"id": document_id},
    )
    graph.query(
        f"MATCH (d:{PDF_DOCUMENT_LABEL} {{id: $id}}) DETACH DELETE d",
        {"id": document_id},
    )


def chunk_metadata(document_id: str, name: str, count: int) -> list[dict]:
    """Per-chunk metadata, which Neo4jVector writes as node properties."""
    return [
        {DOCUMENT_ID_PROPERTY: document_id, "document_name": name}
        for _ in range(count)
    ]
