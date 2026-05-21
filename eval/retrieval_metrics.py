from eval.dataset import EvalSample


def _is_relevant(chunk: str, ground_truth: str) -> bool:
    """Substring match (case-insensitive). Simple but deterministic — no LLM needed."""
    return ground_truth.lower()[:80] in chunk.lower()


def hit_rate(samples: list[EvalSample]) -> float:
    """Fraction of questions where at least one retrieved chunk contains the ground truth."""
    if not samples:
        return 0.0
    hits = sum(
        1
        for s in samples
        if any(_is_relevant(ctx, s["ground_truth"]) for ctx in s["contexts"])
    )
    return hits / len(samples)


def mrr(samples: list[EvalSample]) -> float:
    """Mean Reciprocal Rank — average 1/rank of first relevant chunk."""
    if not samples:
        return 0.0
    reciprocal_ranks: list[float] = []
    for s in samples:
        rank = next(
            (i + 1 for i, ctx in enumerate(s["contexts"]) if _is_relevant(ctx, s["ground_truth"])),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_retrieval_metrics(samples: list[EvalSample]) -> dict[str, float]:
    return {
        "hit_rate": hit_rate(samples),
        "mrr": mrr(samples),
    }
