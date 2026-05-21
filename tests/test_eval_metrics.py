import pytest

from eval.retrieval_metrics import compute_retrieval_metrics, hit_rate, mrr


def _mock_embed(texts: list[str]) -> list[list[float]]:
    """
    Deterministic mock embedder. Each unique string gets a unique orthogonal
    unit vector within the batch. Identical strings always produce cosine = 1.0;
    any two distinct strings produce cosine = 0.0.
    """
    unique = list(dict.fromkeys(texts))
    dim = len(unique)
    index = {t: i for i, t in enumerate(unique)}
    result = []
    for text in texts:
        v = [0.0] * dim
        v[index[text]] = 1.0
        result.append(v)
    return result


def _sample(contexts: list[str], ground_truth: str) -> dict:
    return {"question": "q", "ground_truth": ground_truth, "contexts": contexts, "answer": "a"}


# ── hit_rate ─────────────────────────────────────────────────────────────────

def test_hit_rate_all_hit():
    samples = [_sample(["relevant"], "relevant") for _ in range(5)]
    assert hit_rate(samples, embed_fn=_mock_embed) == 1.0


def test_hit_rate_none_hit():
    samples = [_sample(["unrelated"], "ground truth") for _ in range(5)]
    assert hit_rate(samples, embed_fn=_mock_embed) == 0.0


def test_hit_rate_partial():
    samples = [
        _sample(["relevant"], "relevant"),
        _sample(["unrelated"], "relevant"),
    ]
    assert hit_rate(samples, embed_fn=_mock_embed) == 0.5


# ── mrr ──────────────────────────────────────────────────────────────────────

def test_mrr_first_rank():
    samples = [_sample(["relevant", "other"], "relevant")]
    assert mrr(samples, embed_fn=_mock_embed) == 1.0


def test_mrr_second_rank():
    samples = [_sample(["miss", "relevant"], "relevant")]
    assert mrr(samples, embed_fn=_mock_embed) == 0.5


def test_mrr_no_hit():
    samples = [_sample(["miss", "also miss"], "relevant")]
    assert mrr(samples, embed_fn=_mock_embed) == 0.0


# ── compute_retrieval_metrics ─────────────────────────────────────────────────

def test_compute_returns_both_metrics():
    samples = [_sample(["relevant"], "relevant")]
    result = compute_retrieval_metrics(samples, embed_fn=_mock_embed)
    assert "hit_rate" in result and "mrr" in result


def test_compute_single_embed_call():
    """compute_retrieval_metrics should make one embed call, not two."""
    call_count = {"n": 0}

    def counting_embed(texts):
        call_count["n"] += 1
        return _mock_embed(texts)

    samples = [_sample(["relevant"], "relevant")]
    compute_retrieval_metrics(samples, embed_fn=counting_embed)
    assert call_count["n"] == 1


# ── edge cases ────────────────────────────────────────────────────────────────

def test_empty_samples():
    assert hit_rate([], embed_fn=_mock_embed) == 0.0
    assert mrr([], embed_fn=_mock_embed) == 0.0


def test_multiple_contexts_first_relevant():
    samples = [_sample(["relevant", "noise", "noise"], "relevant")]
    assert mrr(samples, embed_fn=_mock_embed) == 1.0


def test_multiple_contexts_last_relevant():
    samples = [_sample(["noise", "noise", "relevant"], "relevant")]
    assert mrr(samples, embed_fn=_mock_embed) == pytest.approx(1 / 3)

