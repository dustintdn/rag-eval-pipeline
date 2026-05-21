from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.retrievers import BaseRetriever

from config import settings
from ingest.embedder import get_vectorstore


def get_reranking_retriever(top_n: int | None = None) -> ContextualCompressionRetriever:
    """
    Wraps the configured base retriever (hybrid if enabled, else Chroma)
    with a Cohere rerank compressor. Fetches reranker_fetch_k candidates,
    reranks, and returns the top top_n.
    """
    if settings.enable_hybrid_retrieval:
        from retriever.hybrid import get_hybrid_retriever
        base_retriever = get_hybrid_retriever(top_k=settings.reranker_fetch_k)
    else:
        base_retriever = get_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.reranker_fetch_k},
        )
    compressor = CohereRerank(
        cohere_api_key=settings.cohere_api_key,
        model=settings.reranker_model,
        top_n=top_n or settings.reranker_top_n,
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
