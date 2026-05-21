from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = ""

    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "rag_docs"

    chunk_size: int = 512
    chunk_overlap: int = 64

    top_k: int = 4

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"


settings = Settings()
