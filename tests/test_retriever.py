"""
Retriever tests use a mock Chroma vectorstore to avoid hitting OpenAI or
writing to disk. We patch at the embedder level so the rest of the stack
runs as normal.
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def _mock_vectorstore(docs: list[Document]):
    vs = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    vs.as_retriever.return_value = retriever
    return vs


def test_retrieve_returns_documents():
    fake_docs = [
        Document(page_content="chunk one", metadata={"source_file": "a.txt"}),
        Document(page_content="chunk two", metadata={"source_file": "a.txt"}),
    ]
    with patch("retriever.retriever.get_vectorstore", return_value=_mock_vectorstore(fake_docs)):
        from retriever.retriever import retrieve
        results = retrieve("test question", top_k=2)

    assert len(results) == 2
    assert results[0].page_content == "chunk one"


def test_retrieve_respects_top_k():
    fake_docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(4)]
    mock_vs = _mock_vectorstore(fake_docs)

    with patch("retriever.retriever.get_vectorstore", return_value=mock_vs):
        from retriever.retriever import get_retriever
        get_retriever(top_k=4)

    mock_vs.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
