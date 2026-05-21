from eval.retrieval_metrics import compute_retrieval_metrics, hit_rate, mrr


def _sample(contexts: list[str], ground_truth: str) -> dict:
    return {
        "question": "q",
        "ground_truth": ground_truth,
        "contexts": contexts,
        "answer": "a",
    }


def test_hit_rate_all_hit():
    samples = [_sample(["the answer is 42"], "the answer is 42") for _ in range(5)]
    assert hit_rate(samples) == 1.0


def test_hit_rate_none_hit():
    samples = [_sample(["unrelated text"], "ground truth here") for _ in range(5)]
    assert hit_rate(samples) == 0.0


def test_hit_rate_partial():
    samples = [
        _sample(["the answer is 42"], "the answer is 42"),
        _sample(["unrelated"], "the answer is 42"),
    ]
    assert hit_rate(samples) == 0.5


def test_mrr_first_rank():
    samples = [_sample(["the answer is 42", "other"], "the answer is 42")]
    assert mrr(samples) == 1.0


def test_mrr_second_rank():
    samples = [_sample(["miss", "the answer is 42"], "the answer is 42")]
    assert mrr(samples) == 0.5


def test_mrr_no_hit():
    samples = [_sample(["miss", "also miss"], "the answer is 42")]
    assert mrr(samples) == 0.0


def test_compute_returns_both_metrics():
    samples = [_sample(["the answer is 42"], "the answer is 42")]
    result = compute_retrieval_metrics(samples)
    assert "hit_rate" in result and "mrr" in result


def test_empty_samples():
    assert hit_rate([]) == 0.0
    assert mrr([]) == 0.0
