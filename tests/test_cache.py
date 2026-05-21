import hashlib
from unittest.mock import MagicMock

import pytest

from chain.cache import SemanticCache, enable_semantic_cache, get_cache


def _embed_query(text: str) -> list[float]:
    """
    Stable hash-based unit vector. Each unique string maps to its own dimension,
    so identical strings → cosine = 1.0, distinct strings → cosine = 0.0.
    Uses 128 dimensions; collision probability is negligible for test strings.
    """
    dim = 128
    idx = int(hashlib.md5(text.encode()).hexdigest(), 16) % dim
    v = [0.0] * dim
    v[idx] = 1.0
    return v


def _make_cache(threshold=0.95) -> SemanticCache:
    cache = SemanticCache(similarity_threshold=threshold)
    embed_fn = MagicMock()
    embed_fn.embed_query = _embed_query
    cache._embed_fn = embed_fn
    return cache


def _fake_result(answer="42"):
    from chain.qa_chain import QAResult
    return QAResult(answer=answer, source_documents=[], prompt_version="v1")


# ── lookup ────────────────────────────────────────────────────────────────────

def test_lookup_empty_store_returns_none():
    cache = _make_cache()
    assert cache.lookup("anything") is None


def test_lookup_exact_match_returns_result():
    cache = _make_cache()
    result = _fake_result("hello")
    cache.store("what is RAG?", result)
    hit = cache.lookup("what is RAG?")
    assert hit is not None
    assert hit.answer == "hello"


def test_lookup_miss_returns_none():
    cache = _make_cache()
    cache.store("what is RAG?", _fake_result())
    # Different question → orthogonal vector → cosine = 0 → below threshold
    assert cache.lookup("what is a vector store?") is None


def test_lookup_respects_threshold():
    # With threshold=0.0 every lookup hits; with threshold=1.0 only exact matches hit
    cache_loose = _make_cache(threshold=0.0)
    cache_strict = _make_cache(threshold=1.0)

    cache_loose.store("question A", _fake_result("A"))
    cache_strict.store("question A", _fake_result("A"))

    # "question B" is different → cosine = 0
    assert cache_loose.lookup("question B") is not None  # 0.0 >= 0.0 → hit
    assert cache_strict.lookup("question B") is None     # 0.0 < 1.0 → miss


# ── store + len ───────────────────────────────────────────────────────────────

def test_store_increments_length():
    cache = _make_cache()
    assert len(cache) == 0
    cache.store("q1", _fake_result())
    assert len(cache) == 1
    cache.store("q2", _fake_result())
    assert len(cache) == 2


def test_clear_empties_store():
    cache = _make_cache()
    cache.store("q", _fake_result())
    cache.clear()
    assert len(cache) == 0
    assert cache.lookup("q") is None


# ── module-level singleton ────────────────────────────────────────────────────

def test_enable_semantic_cache_returns_singleton():
    # Reset module state first
    import chain.cache as cache_module
    cache_module._cache = None

    c1 = enable_semantic_cache()
    c2 = enable_semantic_cache()
    assert c1 is c2


def test_get_cache_returns_none_before_enable():
    import chain.cache as cache_module
    cache_module._cache = None
    assert get_cache() is None


def test_get_cache_returns_instance_after_enable():
    import chain.cache as cache_module
    cache_module._cache = None
    enable_semantic_cache()
    assert get_cache() is not None
    cache_module._cache = None  # clean up
