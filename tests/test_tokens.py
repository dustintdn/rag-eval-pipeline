"""Tiktoken-based fallback for token counting."""
from chain.tokens import estimate_tokens


def test_empty_string_returns_zero():
    assert estimate_tokens("", "gpt-4o-mini") == 0


def test_simple_text_token_count_is_reasonable():
    n = estimate_tokens("Hello world", "gpt-4o-mini")
    assert 1 <= n <= 4


def test_longer_text_has_more_tokens():
    short = estimate_tokens("Hello", "gpt-4o-mini")
    long_ = estimate_tokens("Hello " * 100, "gpt-4o-mini")
    assert long_ > short * 50


def test_unknown_model_falls_back_to_cl100k():
    # Shouldn't raise; should return a non-zero count.
    assert estimate_tokens("hello", "fictional-model-7000") > 0
