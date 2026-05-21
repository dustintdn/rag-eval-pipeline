"""
Hybrid retriever tests verify the BM25+dense ensemble construction and
the empty-index fallback. We mock the vectorstore so tests run without
hitting OpenAI or persisting anything.
"""
from unittest.mock import MagicMock, patch

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def _mock_vectorstore(documents: list[str]):
    vs = MagicMock()
    vs.get.return_value = {
        "documents": documents,
        "metadatas": [{} for _ in documents],
    }
    # EnsembleRetriever validates that each retriever is a real Runnable.
    # Use a real BM25Retriever as the dense stand-in so construction succeeds.
    real = BM25Retriever.from_documents([Document(page_content="placeholder")])
    vs.as_retriever.return_value = real
    return vs


def test_hybrid_retriever_returns_ensemble_when_index_nonempty():
    docs = ["hello world", "second chunk", "another piece"]
    with patch("retriever.hybrid.get_vectorstore", return_value=_mock_vectorstore(docs)):
        from retriever.hybrid import get_hybrid_retriever
        r = get_hybrid_retriever(top_k=4)
    assert isinstance(r, EnsembleRetriever)
    assert len(r.retrievers) == 2


def test_hybrid_retriever_falls_back_when_index_empty():
    with patch("retriever.hybrid.get_vectorstore", return_value=_mock_vectorstore([])):
        from retriever.hybrid import get_hybrid_retriever
        r = get_hybrid_retriever(top_k=4)
    # Empty corpus: should NOT be an ensemble — falls back to plain dense
    assert not isinstance(r, EnsembleRetriever)


def test_hybrid_retriever_weights_match_config():
    from config import settings
    docs = ["a", "b"]
    original = settings.hybrid_bm25_weight
    settings.hybrid_bm25_weight = 0.6
    try:
        with patch("retriever.hybrid.get_vectorstore", return_value=_mock_vectorstore(docs)):
            from retriever.hybrid import get_hybrid_retriever
            r = get_hybrid_retriever(top_k=4)
        assert r.weights == [0.6, 0.4]
    finally:
        settings.hybrid_bm25_weight = original
