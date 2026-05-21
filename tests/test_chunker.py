from langchain_core.documents import Document

from ingest.chunker import chunk_documents


def _make_doc(text: str, source: str = "test.txt") -> Document:
    return Document(page_content=text, metadata={"source_file": source})


def test_chunk_count_scales_with_length():
    long_text = "word " * 2000
    doc = _make_doc(long_text)
    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1


def test_chunk_metadata_preserved():
    doc = _make_doc("short text", source="myfile.txt")
    chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=0)
    assert all(c.metadata["source_file"] == "myfile.txt" for c in chunks)


def test_chunk_index_is_set():
    long_text = "x " * 1000
    chunks = chunk_documents([_make_doc(long_text)], chunk_size=100, chunk_overlap=0)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_no_empty_chunks():
    doc = _make_doc("Hello world. " * 100)
    chunks = chunk_documents([doc])
    assert all(c.page_content.strip() for c in chunks)
