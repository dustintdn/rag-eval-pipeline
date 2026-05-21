"""
In-process semantic cache for LLM answers.

Stores (embedding, QAResult) pairs in memory. On each query, computes cosine
similarity between the new question embedding and all cached entries. Returns
the cached answer when similarity exceeds the threshold, skipping the LLM call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chain.qa_chain import QAResult


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


@dataclass
class SemanticCache:
    similarity_threshold: float = 0.95
    _store: list[tuple[list[float], "QAResult"]] = field(default_factory=list, repr=False)
    _embed_fn: object = field(default=None, repr=False)

    def _get_embed_fn(self):
        if self._embed_fn is None:
            from langchain_openai import OpenAIEmbeddings
            from config import settings
            self._embed_fn = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
            )
        return self._embed_fn

    def lookup(self, question: str) -> "QAResult | None":
        if not self._store:
            return None
        vec = self._get_embed_fn().embed_query(question)
        best_score, best_result = max(
            ((_cosine(vec, stored_vec), result) for stored_vec, result in self._store),
            key=lambda x: x[0],
        )
        return best_result if best_score >= self.similarity_threshold else None

    def store(self, question: str, result: "QAResult") -> None:
        vec = self._get_embed_fn().embed_query(question)
        self._store.append((vec, result))

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# Module-level singleton; only active when enable_semantic_cache=True in config
_cache: SemanticCache | None = None


def get_cache() -> SemanticCache | None:
    return _cache


def enable_semantic_cache(threshold: float = 0.95) -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(similarity_threshold=threshold)
    return _cache
