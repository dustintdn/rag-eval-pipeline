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
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"Running eval on {dataset_path}…")
    run_id, results = run_eval(dataset_path)
    print(f"\nRun ID: {run_id}")
    print("\nScores:")
    for metric, score in results["scores"].items():
        print(f"  {metric}: {score:.4f}")
    print(f"\nFull results written to eval_logs/{run_id}_results.json")


if __name__ == "__main__":
    main()
