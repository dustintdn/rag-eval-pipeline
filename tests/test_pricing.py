"""Cost-estimation lookup tests."""
import pytest

from config import MODEL_PRICING_PER_1K, estimate_cost_usd


def test_known_model_returns_nonzero_cost():
    cost = estimate_cost_usd("gpt-4o-mini", 1000, 1000)
    rates = MODEL_PRICING_PER_1K["gpt-4o-mini"]
    assert cost == pytest.approx(rates["prompt"] + rates["completion"])


def test_unknown_model_returns_zero():
    assert estimate_cost_usd("some-unreleased-model", 1000, 1000) == 0.0


def test_zero_tokens_returns_zero():
    assert estimate_cost_usd("gpt-4o-mini", 0, 0) == 0.0


def test_cost_scales_linearly():
    a = estimate_cost_usd("gpt-4o-mini", 1000, 1000)
    b = estimate_cost_usd("gpt-4o-mini", 2000, 2000)
    assert abs(b - 2 * a) < 1e-9
