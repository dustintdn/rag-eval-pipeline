"""CLI: python scripts/eval_compare.py <run_a_id> <run_b_id>

Prints a metric-delta table between two eval logs. Exit code is 1 when any
quality metric in run B regresses by more than --threshold (default 0.05)
vs run A; useful as a CI gate.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.runner import EVAL_LOGS_DIR

# Higher = better for these metrics. Latency/cost/tokens trend the opposite way.
QUALITY_METRICS = {
    "hit_rate", "mrr",
    "faithfulness", "answer_relevancy",
    "context_precision", "context_recall",
}


def _load(run_id: str) -> dict:
    path = EVAL_LOGS_DIR / f"{run_id}_results.json"
    if not path.exists():
        print(f"Run not found: {path}", file=sys.stderr)
        sys.exit(2)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Print delta between two eval runs")
    parser.add_argument("run_a", help="Baseline run ID")
    parser.add_argument("run_b", help="Candidate run ID")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Regression threshold for quality metrics (default 0.05)")
    args = parser.parse_args()

    a = _load(args.run_a)
    b = _load(args.run_b)

    if a.get("dataset_version") != b.get("dataset_version"):
        print(f"WARNING: dataset_version differs (A: {a.get('dataset_version')}, "
              f"B: {b.get('dataset_version')})\n", file=sys.stderr)

    scores_a, scores_b = a["scores"], b["scores"]
    metrics = sorted(set(scores_a) | set(scores_b))

    header = f"{'metric':<24} {'run_a':>12} {'run_b':>12} {'delta':>12}"
    print(header)
    print("-" * len(header))

    regressions = []
    for m in metrics:
        va, vb = scores_a.get(m), scores_b.get(m)
        if va is None or vb is None:
            print(f"{m:<24} {str(va):>12} {str(vb):>12} {'-':>12}")
            continue
        delta = vb - va
        print(f"{m:<24} {va:>12.4f} {vb:>12.4f} {delta:>+12.4f}")
        if m in QUALITY_METRICS and delta < -args.threshold:
            regressions.append((m, delta))

    if regressions:
        print("\nRegressions exceeding threshold:")
        for m, d in regressions:
            print(f"  {m}: {d:+.4f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
