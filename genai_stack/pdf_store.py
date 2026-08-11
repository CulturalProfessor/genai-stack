"""PDFs uploaded into a chat, kept side by side instead of one at a time.

The chunks live in the same `pdf_bot` vector index they always did, but each
one now carries the id of the document it came from, so a question can be
answered from one PDF without deleting the others first.

    (:PdfDocument {id, name, chunks, pages, uploaded_at})
    (:PdfBotChunk {text, embedding, document_id, document_name})

    (:Conversation)-[:USES_DOCUMENT]->(:PdfDocument)

There is no relationship between the document and its chunks: those are
written by Neo4jVector, which only knows how to set properties from a
document's metadata. `document_id` is that link, and what retrieval filters on.

The edge from a conversation is what makes a PDF part of a chat rather than a
separate page. A document is shared - the id is content-addressed, so the same
file attached to two chats is embedded once - which is why detaching deletes
the chunks only when the last chat lets go of it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from genai_stack.conversations import CONVERSATION_LABEL, USES_DOCUMENT

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


def attach_document(graph, conversation_id: str, document_id: str) -> None:
    """Make a PDF part of a chat. MERGE, so attaching twice is attaching once."""
    graph.query(
        f"""
        MATCH (c:{CONVERSATION_LABEL} {{id: $conversation_id}})
        MATCH (d:{PDF_DOCUMENT_LABEL} {{id: $document_id}})
        MERGE (c)-[:{USES_DOCUMENT}]->(d)
        """,
        {"conversation_id": conversation_id, "document_id": document_id},
    )


def detach_document(graph, conversation_id: str, document_id: str) -> bool:
    """Take a PDF out of one chat, and out of the store if it was the last.

    Returns whether the document itself went with it. Nothing lists documents
    on their own any more, so one left behind by its last chat is unreachable -
    and its chunks are the expensive part to leave lying around.
    """
    graph.query(
        f"""
        MATCH (:{CONVERSATION_LABEL} {{id: $conversation_id}})
              -[r:{USES_DOCUMENT}]->(:{PDF_DOCUMENT_LABEL} {{id: $document_id}})
        DELETE r
        """,
        {"conversation_id": conversation_id, "document_id": document_id},
    )
    rows = graph.query(
        f"""
        MATCH (:{CONVERSATION_LABEL})-[:{USES_DOCUMENT}]->
              (d:{PDF_DOCUMENT_LABEL} {{id: $document_id}})
        RETURN count(*) AS still_used
        """,
        {"document_id": document_id},
    )
    if rows and rows[0]["still_used"]:
        return False
    delete_document(graph, document_id)
    return True


def documents_for(graph, conversation_id: str) -> list[PdfDocument]:
    """The PDFs attached to one chat, in the order they were uploaded."""
    rows = graph.query(
        f"""
        MATCH (:{CONVERSATION_LABEL} {{id: $id}})-[:{USES_DOCUMENT}]->
              (d:{PDF_DOCUMENT_LABEL})
        RETURN d.id AS id, d.name AS name, d.chunks AS chunks,
               d.pages AS pages, d.uploaded_at AS uploaded_at
        ORDER BY d.uploaded_at ASC
        """,
        {"id": conversation_id},
    )
    return [PdfDocument(**row) for row in rows]


def adopt_legacy_conversations(graph) -> None:
    """Bring chats written before PDFs lived inside a normal chat forward.

    Those carried the document id in a `subject` property and a `kind` telling
    the two pages apart. Both are now an edge, and a chat whose document is
    gone becomes an ordinary chat rather than one that shows up nowhere.
    """
    graph.query(
        f"""
        MATCH (c:{CONVERSATION_LABEL}) WHERE c.subject IS NOT NULL
        MATCH (d:{PDF_DOCUMENT_LABEL} {{id: c.subject}})
        MERGE (c)-[:{USES_DOCUMENT}]->(d)
        """
    )
    graph.query(
        f"""
        MATCH (c:{CONVERSATION_LABEL})
        WHERE c.subject IS NOT NULL OR c.kind IS NOT NULL
        REMOVE c.subject, c.kind
        """
    )


def delete_document(graph, document_id: str) -> None:
    """Remove a PDF and its chunks, whatever is still pointing at it."""
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


def chunk_metadata(document_id: str, name: str, pages: list[int]) -> list[dict]:
    """Per-chunk metadata, which Neo4jVector writes as node properties.

    One page number per chunk, so an answer can cite where in the document it
    came from rather than only which document.
    """
    return [
        {DOCUMENT_ID_PROPERTY: document_id, "document_name": name, "page": page}
        for page in pages
    ]
