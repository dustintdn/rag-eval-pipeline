import json
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from eval.dataset import EvalSample, load_dataset
from eval.ragas_eval import run_ragas
from eval.retrieval_metrics import compute_retrieval_metrics
from prompts.registry import load_prompt

EVAL_LOGS_DIR = Path("eval_logs")


def generate_live_samples(samples: list[EvalSample]) -> list[EvalSample]:
    """Run each question through the live pipeline to populate contexts and answer."""
    from chain.qa_chain import ask

    live: list[EvalSample] = []
    for i, s in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {s['question'][:60]}")
        result = ask(s["question"])
        live.append({
            "question": s["question"],
            "ground_truth": s["ground_truth"],
            "contexts": [doc.page_content for doc in result.source_documents],
            "answer": result.answer,
        })
    return live


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


def run_eval(dataset_path: str | Path, live: bool = False) -> tuple[str, dict]:
    """Run the full eval pipeline and write results. Returns (run_id, results_dict)."""
    samples: list[EvalSample] = load_dataset(dataset_path)

    if live:
        print(f"Generating live answers for {len(samples)} questions…")
        samples = generate_live_samples(samples)

    retrieval_scores = compute_retrieval_metrics(samples)
    ragas_scores = run_ragas(samples)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = {
        "run_id": run_id,
        "dataset": str(dataset_path),
        "config": _config_snapshot(live),
        "scores": {**retrieval_scores, **ragas_scores},
        "per_question": [
            {
                "question": s["question"],
                "answer": s["answer"],
                "ground_truth": s["ground_truth"],
                "num_contexts": len(s["contexts"]),
            }
            for s in samples
        ],
    }

    EVAL_LOGS_DIR.mkdir(exist_ok=True)
    out_path = EVAL_LOGS_DIR / f"{run_id}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return run_id, results
