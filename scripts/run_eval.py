"""CLI: python scripts/run_eval.py --dataset eval/sample_dataset.json"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.runner import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full eval pipeline")
    parser.add_argument(
        "--dataset",
        default="eval/sample_dataset.json",
        help="Path to the eval dataset JSON",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run each question through the live retriever + chain before scoring",
    )
    parser.add_argument("--prompt-version", help="Override PROMPT_VERSION for this run")
    parser.add_argument("--top-k", type=int, help="Override TOP_K for this run")
    rerank = parser.add_mutually_exclusive_group()
    rerank.add_argument("--reranker", dest="reranker", action="store_true", default=None,
                        help="Force-enable the Cohere reranker for this run")
    rerank.add_argument("--no-reranker", dest="reranker", action="store_false",
                        help="Force-disable the Cohere reranker for this run")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    overrides: dict = {}
    if args.prompt_version:
        overrides["prompt_version"] = args.prompt_version
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.reranker is not None:
        overrides["enable_reranker"] = args.reranker

    mode = "live" if args.live else "static"
    print(f"Running eval on {dataset_path} [{mode}]…")
    if overrides:
        print(f"Overrides: {overrides}")
    run_id, results = run_eval(dataset_path, live=args.live, config_overrides=overrides or None)
    print(f"\nRun ID: {run_id}")
    print("\nScores:")
    for metric, score in results["scores"].items():
        print(f"  {metric}: {score:.4f}")
    print(f"\nFull results written to eval_logs/{run_id}_results.json")


if __name__ == "__main__":
    main()
