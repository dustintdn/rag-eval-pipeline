"""
Token counting helpers. Prefer LLM-reported `usage_metadata` (exact);
fall back to a tiktoken-based estimate when the provider doesn't emit
it (some chunk types, older API versions).
"""
from functools import lru_cache

import tiktoken


@lru_cache(maxsize=8)
def _encoder_for(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Unknown model — fall back to the cl100k_base encoding used by
        # gpt-4 and gpt-3.5-turbo families.
        return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str, model: str) -> int:
    """Best-effort token count for the given text/model."""
    if not text:
        return 0
    return len(_encoder_for(model).encode(text))
