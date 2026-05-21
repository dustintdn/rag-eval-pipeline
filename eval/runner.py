import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import estimate_cost_usd, settings
from eval.dataset import EvalSample, load_dataset
from eval.ragas_eval import run_ragas
from eval.retrieval_metrics import compute_retrieval_metrics_detailed
from logger import get_logger
from prompts.registry import load_prompt

EVAL_LOGS_DIR = Path("eval_logs")
logger = get_logger(__name__)


@contextmanager
def _settings_override(overrides: dict | None):
    """Temporarily override fields on the global settings singleton."""
    if not overrides:
        yield
        return
    previous = {k: getattr(settings, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(settings, k, v)
        yield
    finally:
        for k, v in previous.items():
            setattr(settings, k, v)


def generate_live_samples(
    samples: list[EvalSample],
) -> tuple[list[EvalSample], list[float], list[dict | None], list[bool]]:
    """Run each question through the live pipeline.

    Returns (samples, latencies, token_usages, cache_hits).
    """
    from chain.qa_chain import ask

    live: list[EvalSample] = []
    latencies: list[float] = []
    tokens: list[dict | None] = []
    cache_hits: list[bool] = []
    for i, s in enumerate(samples, 1):
        logger.info("[%d/%d] %s", i, len(samples), s["question"][:60])
        start = time.perf_counter()
        result = ask(s["question"])
        latencies.append(time.perf_counter() - start)
        tokens.append(dict(result.token_usage) if result.token_usage else None)
        cache_hits.append(result.from_cache)
        live.append({
            "question": s["question"],
            "ground_truth": s["ground_truth"],
            "contexts": [doc.page_content for doc in result.source_documents],
            "answer": result.answer,
        })
    return live, latencies, tokens, cache_hits


def _config_snapshot(live: bool) -> dict:
    _, prompt_meta = load_prompt(settings.prompt_version)
    return {
        "live_eval": live,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "collection_name": settings.collection_name,
        "prompt_version": prompt_meta["version"],
        "prompt_description": prompt_meta["description"],
        "reranker_enabled": settings.enable_reranker,
        "reranker_model": settings.reranker_model if settings.enable_reranker else None,
        "reranker_fetch_k": settings.reranker_fetch_k if settings.enable_reranker else None,
        "reranker_top_n": settings.reranker_top_n if settings.enable_reranker else None,
    }


def run_eval(
    dataset_path: str | Path,
    live: bool = False,
    config_overrides: dict | None = None,
) -> tuple[str, dict]:
    """Run the full eval pipeline and write results. Returns (run_id, results_dict).

    `config_overrides` temporarily applies settings fields (e.g. `top_k=8`,
    `prompt_version="v2_concise"`, `enable_reranker=True`) for this run only.
    """
    samples: list[EvalSample] = load_dataset(dataset_path)
    latencies: list[float] = []
    tokens: list[dict | None] = []
    cache_hits: list[bool] = []

    with _settings_override(config_overrides):
        if live:
            logger.info("Generating live answers for %d questions", len(samples))
            samples, latencies, tokens, cache_hits = generate_live_samples(samples)

        retrieval_scores, retrieval_per_sample = compute_retrieval_metrics_detailed(samples)
        ragas_scores, ragas_per_sample = run_ragas(samples)
        config_snap = _config_snapshot(live)

    per_question_costs = [
        estimate_cost_usd(settings.llm_model, t["prompt"], t["completion"])
        if t is not None else 0.0
        for t in tokens
    ]

    aggregate_scores: dict = {**retrieval_scores, **ragas_scores}
    if latencies:
        aggregate_scores["mean_latency_seconds"] = sum(latencies) / len(latencies)
    valid_token_totals = [t["total"] for t in tokens if t is not None]
    if valid_token_totals:
        aggregate_scores["mean_total_tokens"] = sum(valid_token_totals) / len(valid_token_totals)
    if per_question_costs and any(c > 0 for c in per_question_costs):
        aggregate_scores["total_cost_usd"] = sum(per_question_costs)
    if cache_hits:
        aggregate_scores["cache_hit_rate"] = sum(cache_hits) / len(cache_hits)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = {
        "run_id": run_id,
        "dataset": str(dataset_path),
        "config": config_snap,
        "scores": aggregate_scores,
        "per_question": [
            {
                "question": s["question"],
                "answer": s["answer"],
                "ground_truth": s["ground_truth"],
                "num_contexts": len(s["contexts"]),
                **({"latency_seconds": latencies[i]} if i < len(latencies) else {}),
                **({"tokens": tokens[i]} if i < len(tokens) and tokens[i] is not None else {}),
                **({"cost_usd": per_question_costs[i]} if i < len(per_question_costs) and per_question_costs[i] > 0 else {}),
                **({"from_cache": cache_hits[i]} if i < len(cache_hits) else {}),
                "scores": {
                    **(retrieval_per_sample[i] if i < len(retrieval_per_sample) else {}),
                    **(ragas_per_sample[i] if i < len(ragas_per_sample) else {}),
                },
            }
            for i, s in enumerate(samples)
        ],
    }

    EVAL_LOGS_DIR.mkdir(exist_ok=True)
    out_path = EVAL_LOGS_DIR / f"{run_id}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return run_id, results
