"""
Embedder tests verify that ingestion is idempotent — re-ingesting the same
chunk should upsert rather than duplicate, so we pass stable IDs to Chroma.
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from ingest.embedder import _chunk_id, embed_and_store


def test_chunk_id_uses_source_and_index():
    chunk = Document(page_content="x", metadata={"source_file": "a.txt", "chunk_index": 3})
    assert _chunk_id(chunk) == "a.txt::3"


def test_chunk_id_includes_page_number_when_present():
    chunk = Document(
        page_content="x",
        metadata={"source_file": "doc.pdf", "chunk_index": 2, "page_number": 5},
    )
    assert _chunk_id(chunk) == "doc.pdf::5::2"


def test_chunk_id_is_stable_across_runs():
    chunk = Document(page_content="x", metadata={"source_file": "a.txt", "chunk_index": 0})
    assert _chunk_id(chunk) == _chunk_id(chunk)


def test_embed_and_store_passes_stable_ids():
    chunks = [
        Document(page_content="hello", metadata={"source_file": "a.txt", "chunk_index": 0}),
        Document(page_content="world", metadata={"source_file": "a.txt", "chunk_index": 1}),
    ]
    mock_vs = MagicMock()
    with patch("ingest.embedder.get_vectorstore", return_value=mock_vs):
        count = embed_and_store(chunks)

    assert count == 2
    _, kwargs = mock_vs.add_documents.call_args
    assert kwargs["ids"] == ["a.txt::0", "a.txt::1"]
