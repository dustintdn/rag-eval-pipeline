from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.retrievers import BaseRetriever

from config import settings
from ingest.embedder import get_vectorstore


def get_reranking_retriever() -> ContextualCompressionRetriever:
    """
    Wraps the Chroma retriever with a Cohere rerank compressor.
    Fetches reranker_fetch_k candidates, reranks, and returns the top reranker_top_n.
    """
    base_retriever = get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.reranker_fetch_k},
    )
    compressor = CohereRerank(
        cohere_api_key=settings.cohere_api_key,
        model=settings.reranker_model,
        top_n=settings.reranker_top_n,
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
