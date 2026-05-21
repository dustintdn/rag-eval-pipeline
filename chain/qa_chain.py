from dataclasses import dataclass, field
from typing import TypedDict

from langchain.chains import RetrievalQA
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import settings
from chain.cache import enable_semantic_cache
from prompts.registry import DEFAULT_VERSION, load_prompt
from retriever.retriever import get_retriever
from retriever.reranker import get_reranking_retriever


class TokenUsage(TypedDict):
    prompt: int
    completion: int
    total: int


@dataclass
class QAResult:
    answer: str
    source_documents: list[Document]
    prompt_version: str = field(default=DEFAULT_VERSION)
    token_usage: TokenUsage | None = None


def build_chain(top_k: int | None = None, prompt_version: str | None = None) -> tuple[RetrievalQA, str]:
    """Return (chain, prompt_version_used)."""
    if settings.enable_semantic_cache:
        enable_semantic_cache()
    version = prompt_version or settings.prompt_version
    prompt, _ = load_prompt(version)

    retriever = (
        get_reranking_retriever(top_n=top_k)
        if settings.enable_reranker and settings.cohere_api_key
        else get_retriever(top_k)
    )

    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain, version


def ask(question: str, top_k: int | None = None, prompt_version: str | None = None) -> QAResult:
    from chain.cache import get_cache
    cache = get_cache()
    if cache is not None:
        hit = cache.lookup(question)
        if hit is not None:
            return hit

    chain, version = build_chain(top_k, prompt_version)
    with get_openai_callback() as cb:
        result = chain.invoke({"query": question})
    qa_result = QAResult(
        answer=result["result"],
        source_documents=result["source_documents"],
        prompt_version=version,
        token_usage={
            "prompt": cb.prompt_tokens,
            "completion": cb.completion_tokens,
            "total": cb.total_tokens,
        },
    )
    if cache is not None:
        cache.store(question, qa_result)
    return qa_result
