import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.pdf_store import (
    PdfDocument,
    chunk_metadata,
    delete_document,
    document_id,
    list_documents,
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


def test_list_documents_maps_rows():
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
    assert list_documents(graph) == [
        PdfDocument(
            id="doc-1",
            name="paper.pdf",
            chunks=12,
            pages=3,
            uploaded_at="2026-01-01T00:00:00+00:00",
        )
    ]


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
