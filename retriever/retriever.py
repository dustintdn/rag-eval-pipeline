from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from ingest.embedder import get_vectorstore


def get_retriever(top_k: int | None = None):
    vectorstore: Chroma = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k or settings.top_k},
    )


def retrieve(question: str, top_k: int | None = None) -> list[Document]:
    return get_retriever(top_k).invoke(question)
