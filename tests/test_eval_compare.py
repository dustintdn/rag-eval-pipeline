"""
Smoke tests for scripts/eval_compare.py — verifies exit codes and that
regressions on quality metrics are detected, while tolerable shifts on
non-quality metrics (latency, cost) don't trip the gate.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "eval_compare.py"


def _write_run(path: Path, run_id: str, scores: dict, dataset_version: str = "v1") -> None:
    path.write_text(json.dumps({
        "run_id": run_id,
        "dataset": "eval/sample_dataset.json",
        "dataset_version": dataset_version,
        "config": {},
        "scores": scores,
    }))


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd, capture_output=True, text=True,
    )


def test_exit_zero_when_no_regression(tmp_path):
    logs = tmp_path / "eval_logs"
    logs.mkdir()
    _write_run(logs / "A_results.json", "A", {"faithfulness": 0.8, "hit_rate": 0.6})
    _write_run(logs / "B_results.json", "B", {"faithfulness": 0.85, "hit_rate": 0.7})
    r = _run("A", "B", cwd=tmp_path)
    assert r.returncode == 0, r.stderr


def test_exit_one_on_regression(tmp_path):
    logs = tmp_path / "eval_logs"
    logs.mkdir()
    _write_run(logs / "A_results.json", "A", {"faithfulness": 0.9, "hit_rate": 0.6})
    _write_run(logs / "B_results.json", "B", {"faithfulness": 0.7, "hit_rate": 0.6})
    r = _run("A", "B", "--threshold", "0.05", cwd=tmp_path)
    assert r.returncode == 1
    assert "faithfulness" in r.stdout


def test_latency_regression_does_not_block(tmp_path):
    """Latency is not a quality metric; growth should NOT trip the gate."""
    logs = tmp_path / "eval_logs"
    logs.mkdir()
    _write_run(logs / "A_results.json", "A", {"faithfulness": 0.9, "mean_latency_seconds": 0.5})
    _write_run(logs / "B_results.json", "B", {"faithfulness": 0.9, "mean_latency_seconds": 2.0})
    r = _run("A", "B", cwd=tmp_path)
    assert r.returncode == 0, r.stderr


def test_dataset_version_mismatch_warns(tmp_path):
    logs = tmp_path / "eval_logs"
    logs.mkdir()
    _write_run(logs / "A_results.json", "A", {"faithfulness": 0.9}, dataset_version="v1")
    _write_run(logs / "B_results.json", "B", {"faithfulness": 0.9}, dataset_version="v2")
    r = _run("A", "B", cwd=tmp_path)
    assert r.returncode == 0
    assert "dataset_version differs" in r.stderr


def test_missing_run_returns_2(tmp_path):
    logs = tmp_path / "eval_logs"
    logs.mkdir()
    r = _run("missing_a", "missing_b", cwd=tmp_path)
    assert r.returncode == 2
