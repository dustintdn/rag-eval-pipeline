import math
from typing import Callable

from eval.dataset import EvalSample

EmbedFn = Callable[[list[str]], list[list[float]]]
DEFAULT_THRESHOLD = 0.75


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def _default_embed_fn() -> EmbedFn:
    from langchain_openai import OpenAIEmbeddings
    from config import settings
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    ).embed_documents


def _relevance_matrix(
    samples: list[EvalSample],
    embed_fn: EmbedFn,
    threshold: float,
) -> list[list[bool]]:
    """Embed all texts in one batch; return per-sample per-chunk relevance booleans."""
    ground_truths = [s["ground_truth"] for s in samples]
    flat_contexts = [ctx for s in samples for ctx in s["contexts"]]

    all_vecs = embed_fn(ground_truths + flat_contexts)
    gt_vecs = all_vecs[: len(ground_truths)]
    ctx_vecs = all_vecs[len(ground_truths) :]

    matrix: list[list[bool]] = []
    offset = 0
    for i, s in enumerate(samples):
        n = len(s["contexts"])
        matrix.append([
            _cosine(gt_vecs[i], ctx_vecs[offset + j]) >= threshold
            for j in range(n)
        ])
        offset += n
    return matrix


def hit_rate(
    samples: list[EvalSample],
    embed_fn: EmbedFn | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    if not samples:
        return 0.0
    relevances = _relevance_matrix(samples, embed_fn or _default_embed_fn(), threshold)
    return sum(1 for row in relevances if any(row)) / len(samples)


def mrr(
    samples: list[EvalSample],
    embed_fn: EmbedFn | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    if not samples:
        return 0.0
    relevances = _relevance_matrix(samples, embed_fn or _default_embed_fn(), threshold)
    rr = [
        1 / next((i + 1 for i, r in enumerate(row) if r), 0) if any(row) else 0.0
        for row in relevances
    ]
    return sum(rr) / len(rr)


def compute_retrieval_metrics(
    samples: list[EvalSample],
    embed_fn: EmbedFn | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    if not samples:
        return {"hit_rate": 0.0, "mrr": 0.0}
    # One embed call shared across both metrics
    relevances = _relevance_matrix(samples, embed_fn or _default_embed_fn(), threshold)
    hits = sum(1 for row in relevances if any(row))
    rr = [
        1 / next((i + 1 for i, r in enumerate(row) if r), 0) if any(row) else 0.0
        for row in relevances
    ]
    return {
        "hit_rate": hits / len(samples),
        "mrr": sum(rr) / len(rr),
    }


def compute_retrieval_metrics_detailed(
    samples: list[EvalSample],
    embed_fn: EmbedFn | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Return (means, per_sample). per_sample[i] = {hit, reciprocal_rank}."""
    if not samples:
        return {"hit_rate": 0.0, "mrr": 0.0}, []
    relevances = _relevance_matrix(samples, embed_fn or _default_embed_fn(), threshold)
    per_sample = []
    for row in relevances:
        hit = any(row)
        rr_val = 1 / next((i + 1 for i, r in enumerate(row) if r), 1) if hit else 0.0
        per_sample.append({"hit": float(hit), "reciprocal_rank": rr_val})
    means = {
        "hit_rate": sum(s["hit"] for s in per_sample) / len(per_sample),
        "mrr": sum(s["reciprocal_rank"] for s in per_sample) / len(per_sample),
    }
    return means, per_sample
