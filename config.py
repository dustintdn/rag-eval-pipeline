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

    prompt_version: str = "v1_cite_sources"
    enable_semantic_cache: bool = False

    cohere_api_key: str = ""
    enable_reranker: bool = False
    reranker_model: str = "rerank-english-v3.0"
    reranker_top_n: int = 4
    reranker_fetch_k: int = 10

    enable_hybrid_retrieval: bool = False
    hybrid_bm25_weight: float = 0.4

    api_token: str = ""


settings = Settings()


# Per-1K-token USD pricing. Values mirror OpenAI's published rates as of
# early 2026; update alongside model swaps. Unknown models price as 0.
MODEL_PRICING_PER_1K = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = MODEL_PRICING_PER_1K.get(model)
    if not rates:
        return 0.0
    return (prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]) / 1000
