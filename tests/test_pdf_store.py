import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.pdf_store import (
    PdfDocument,
    adopt_legacy_conversations,
    attach_document,
    chunk_metadata,
    delete_document,
    detach_document,
    document_id,
    documents_for,
    register_document,
)


class FakeGraph:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
        self.calls = []

    def query(self, cypher, params=None):
        self.calls.append((" ".join(cypher.split()), params or {}))
        return self.rows


def test_document_id_is_stable_for_the_same_file():
    assert document_id("paper.pdf", b"content") == document_id("paper.pdf", b"content")


def test_document_id_changes_with_the_content():
    assert document_id("paper.pdf", b"one") != document_id("paper.pdf", b"two")


def test_document_id_changes_with_the_name():
    """The same bytes saved twice are two entries in a list you pick from."""
    assert document_id("a.pdf", b"content") != document_id("b.pdf", b"content")


def test_register_document_merges_so_a_re_upload_updates():
    graph = FakeGraph()
    register_document(graph, "doc-1", "paper.pdf", chunks=12, pages=3)

    cypher, params = graph.calls[0]
    assert "MERGE (d:PdfDocument {id: $id})" in cypher
    assert params["name"] == "paper.pdf"
    assert params["chunks"] == 12
    assert params["pages"] == 3
    assert params["uploaded_at"]


def test_documents_for_maps_rows():
    graph = FakeGraph(
        rows=[
            {
                "id": "doc-1",
                "name": "paper.pdf",
                "chunks": 12,
                "pages": 3,
                "uploaded_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    )
    assert documents_for(graph, "conv-1") == [
        PdfDocument(
            id="doc-1",
            name="paper.pdf",
            chunks=12,
            pages=3,
            uploaded_at="2026-01-01T00:00:00+00:00",
        )
    ]


def test_attach_document_merges_so_attaching_twice_is_once():
    graph = FakeGraph()
    attach_document(graph, "conv-1", "doc-1")

    cypher, params = graph.calls[0]
    assert "MERGE (c)-[:USES_DOCUMENT]->(d)" in cypher
    assert params == {"conversation_id": "conv-1", "document_id": "doc-1"}


def test_detach_document_keeps_one_another_chat_still_uses():
    graph = FakeGraph(rows=[{"still_used": 1}])
    deleted = detach_document(graph, "conv-1", "doc-1")

    assert deleted is False
    assert len(graph.calls) == 2  # the edge, then the count - nothing dropped
    assert "DELETE r" in graph.calls[0][0]


def test_detach_document_drops_one_no_chat_is_left_holding():
    graph = FakeGraph(rows=[{"still_used": 0}])
    deleted = detach_document(graph, "conv-1", "doc-1")

    assert deleted is True
    # The edge, the count, then delete_document's two queries.
    assert [call[0] for call in graph.calls[2:]] == [
        "MATCH (c:PdfBotChunk {document_id: $id}) DETACH DELETE c",
        "MATCH (d:PdfDocument {id: $id}) DETACH DELETE d",
    ]


def test_adopt_legacy_conversations_turns_a_subject_into_an_edge():
    graph = FakeGraph()
    adopt_legacy_conversations(graph)

    edge, cleanup = (call[0] for call in graph.calls)
    assert "MATCH (d:PdfDocument {id: c.subject})" in edge
    assert "MERGE (c)-[:USES_DOCUMENT]->(d)" in edge
    assert "REMOVE c.subject, c.kind" in cleanup


def test_delete_document_removes_chunks_and_the_document():
    graph = FakeGraph()
    delete_document(graph, "doc-1")

    chunks, document = graph.calls
    assert "MATCH (c:PdfBotChunk {document_id: $id})" in chunks[0]
    assert "MATCH (d:PdfDocument {id: $id})" in document[0]
    assert chunks[1] == {"id": "doc-1"} == document[1]


def test_chunk_metadata_tags_every_chunk_with_its_document():
    metadata = chunk_metadata("doc-1", "paper.pdf", 3)

    assert len(metadata) == 3
    assert all(entry["document_id"] == "doc-1" for entry in metadata)
    assert all(entry["document_name"] == "paper.pdf" for entry in metadata)
