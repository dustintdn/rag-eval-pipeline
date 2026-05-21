from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import settings


def _get_embedding_fn() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=_get_embedding_fn(),
    )


def embed_and_store(chunks: list[Document]) -> int:
    """Embed chunks and upsert into Chroma. Returns the number of chunks added."""
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)
