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


def _chunk_id(chunk: Document) -> str:
    """Stable ID from source_file + chunk_index so re-ingestion upserts."""
    source = chunk.metadata.get("source_file", "unknown")
    index = chunk.metadata.get("chunk_index", 0)
    page = chunk.metadata.get("page_number")
    return f"{source}::{page}::{index}" if page is not None else f"{source}::{index}"


def embed_and_store(chunks: list[Document]) -> int:
    """Embed chunks and upsert into Chroma. Returns the number of chunks added."""
    vectorstore = get_vectorstore()
    ids = [_chunk_id(c) for c in chunks]
    vectorstore.add_documents(chunks, ids=ids)
    return len(chunks)
