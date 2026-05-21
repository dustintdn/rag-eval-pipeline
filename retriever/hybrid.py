"""
Hybrid retrieval: union of BM25 (lexical) and dense (Chroma) results via
EnsembleRetriever. The BM25 retriever indexes the same collection's
documents in memory at construction time, so it only sees what has been
ingested so far. For tiny corpora (sample docs) this is fine; for larger
corpora it's a fixed cost paid once per query.
"""
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import settings
from ingest.embedder import get_vectorstore


def _all_documents() -> list[Document]:
    """Materialise every chunk in the current Chroma collection."""
    raw = get_vectorstore().get()
    return [
        Document(page_content=c, metadata=m or {})
        for c, m in zip(raw.get("documents") or [], raw.get("metadatas") or [])
    ]


def get_hybrid_retriever(top_k: int | None = None) -> BaseRetriever:
    k = top_k or settings.top_k
    docs = _all_documents()
    if not docs:
        # Empty index: BM25 raises on empty corpus. Fall back to plain dense.
        return get_vectorstore().as_retriever(search_kwargs={"k": k})
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    dense = get_vectorstore().as_retriever(search_kwargs={"k": k})
    bm25_weight = settings.hybrid_bm25_weight
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[bm25_weight, 1.0 - bm25_weight],
    )
