import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api.main import app

client = TestClient(app)


# ── /ingest ───────────────────────────────────────────────────────────────────

def _txt_upload(content: str = "hello world", filename: str = "test.txt"):
    return {"file": (filename, BytesIO(content.encode()), "text/plain")}


@patch("api.main.embed_and_store", return_value=3)
@patch("api.main.chunk_documents", return_value=["c1", "c2", "c3"])
@patch("api.main.load_file", return_value=[MagicMock()])
def test_ingest_txt(mock_load, mock_chunk, mock_embed):
    resp = client.post("/ingest", files=_txt_upload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "test.txt"
    assert body["chunks_added"] == 3


@patch("api.main.embed_and_store", return_value=2)
@patch("api.main.chunk_documents", return_value=["c1", "c2"])
@patch("api.main.load_file", return_value=[MagicMock()])
def test_ingest_md(mock_load, mock_chunk, mock_embed):
    resp = client.post("/ingest", files={"file": ("doc.md", BytesIO(b"# hi"), "text/markdown")})
    assert resp.status_code == 200
    assert resp.json()["filename"] == "doc.md"


def test_ingest_unsupported_type():
    resp = client.post("/ingest", files={"file": ("data.csv", BytesIO(b"a,b"), "text/csv")})
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


# ── /query ────────────────────────────────────────────────────────────────────

def _mock_qa_result(answer: str = "42", version: str = "v1_cite_sources"):
    from chain.qa_chain import QAResult
    doc = Document(page_content="chunk content", metadata={"source_file": "doc.txt"})
    return QAResult(answer=answer, source_documents=[doc], prompt_version=version)


@patch("api.main.ask", return_value=_mock_qa_result())
def test_query_returns_answer(mock_ask):
    resp = client.post("/query", json={"question": "What is RAG?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "42"
    assert body["prompt_version"] == "v1_cite_sources"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["content"] == "chunk content"


@patch("api.main.ask", return_value=_mock_qa_result(version="v2_concise"))
def test_query_respects_prompt_version(mock_ask):
    resp = client.post("/query", json={"question": "What is RAG?", "prompt_version": "v2_concise"})
    assert resp.status_code == 200
    assert resp.json()["prompt_version"] == "v2_concise"
    _, kwargs = mock_ask.call_args
    assert kwargs.get("prompt_version") == "v2_concise"


@patch("api.main.ask", return_value=_mock_qa_result())
def test_query_passes_top_k(mock_ask):
    resp = client.post("/query", json={"question": "What is RAG?", "top_k": 7})
    assert resp.status_code == 200
    _, kwargs = mock_ask.call_args
    assert kwargs.get("top_k") == 7


# ── /eval/run ─────────────────────────────────────────────────────────────────

@patch("api.main.run_eval", return_value=("20260101T000000Z", {}))
def test_eval_run_returns_run_id(mock_run):
    resp = client.post("/eval/run")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "20260101T000000Z"


@patch("api.main.run_eval", return_value=("20260101T000000Z", {}))
def test_eval_run_passes_live_flag(mock_run):
    resp = client.post("/eval/run", json={"live": True})
    assert resp.status_code == 200
    assert resp.json()["live"] is True
    _, kwargs = mock_run.call_args
    assert kwargs.get("live") is True


@patch("api.main.DEFAULT_DATASET", Path("nonexistent_dataset.json"))
def test_eval_run_missing_dataset():
    resp = client.post("/eval/run")
    assert resp.status_code == 404


def test_eval_run_custom_dataset_missing():
    resp = client.post("/eval/run", json={"dataset": "no/such/file.json"})
    assert resp.status_code == 404


@patch("api.main.run_eval", return_value=("20260101T000000Z", {}))
def test_eval_run_forwards_overrides(mock_run):
    resp = client.post("/eval/run", json={
        "prompt_version": "v2_concise",
        "top_k": 8,
        "enable_reranker": True,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["overrides"] == {
        "prompt_version": "v2_concise",
        "top_k": 8,
        "enable_reranker": True,
    }
    _, kwargs = mock_run.call_args
    assert kwargs["config_overrides"] == {
        "prompt_version": "v2_concise",
        "top_k": 8,
        "enable_reranker": True,
    }


# ── /ingest/batch ─────────────────────────────────────────────────────────────

@patch("api.main.embed_and_store", side_effect=[3, 2])
@patch("api.main.chunk_documents", side_effect=[["c1", "c2", "c3"], ["c1", "c2"]])
@patch("api.main.load_file", return_value=[MagicMock()])
def test_ingest_batch_multiple_files(mock_load, mock_chunk, mock_embed):
    resp = client.post(
        "/ingest/batch",
        files=[
            ("files", ("a.txt", BytesIO(b"hello"), "text/plain")),
            ("files", ("b.md", BytesIO(b"# hi"), "text/markdown")),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["files"]) == 2
    assert body["total_chunks_added"] == 5


def test_ingest_batch_rejects_unsupported_in_list():
    resp = client.post(
        "/ingest/batch",
        files=[
            ("files", ("a.txt", BytesIO(b"hello"), "text/plain")),
            ("files", ("c.csv", BytesIO(b"a,b"), "text/csv")),
        ],
    )
    assert resp.status_code == 400


# ── /eval/results/{run_id} ────────────────────────────────────────────────────

def test_eval_results_returns_json(tmp_path):
    fixture = {"run_id": "test123", "scores": {"hit_rate": 0.9}}
    result_file = tmp_path / "test123_results.json"
    result_file.write_text(json.dumps(fixture))

    with patch("api.main.EVAL_LOGS_DIR", tmp_path):
        resp = client.get("/eval/results/test123")

    assert resp.status_code == 200
    assert resp.json()["scores"]["hit_rate"] == 0.9


def test_eval_results_not_found(tmp_path):
    with patch("api.main.EVAL_LOGS_DIR", tmp_path):
        resp = client.get("/eval/results/doesnotexist")
    assert resp.status_code == 404


# ── /prompts ──────────────────────────────────────────────────────────────────

def test_list_prompts():
    resp = client.get("/prompts")
    assert resp.status_code == 200
    body = resp.json()
    assert "versions" in body
    assert "active" in body
    assert "v1_cite_sources" in body["versions"]
    assert "v2_concise" in body["versions"]


# ── /eval/datasets ────────────────────────────────────────────────────────────

def test_list_eval_datasets(tmp_path):
    (tmp_path / "small.json").write_text(json.dumps([
        {"question": "q", "ground_truth": "g", "contexts": ["c"], "answer": "a"},
    ]))
    with patch("api.main.DATASETS_DIR", tmp_path):
        resp = client.get("/eval/datasets")
    body = resp.json()
    assert resp.status_code == 200
    names = [d["name"] for d in body["datasets"]]
    assert "small" in names
    assert body["datasets"][0]["samples"] == 1


def test_get_eval_dataset(tmp_path):
    payload = [{"question": "q", "ground_truth": "g", "contexts": ["c"], "answer": "a"}]
    (tmp_path / "mini.json").write_text(json.dumps(payload))
    with patch("api.main.DATASETS_DIR", tmp_path):
        resp = client.get("/eval/datasets/mini")
    assert resp.status_code == 200
    assert resp.json()["samples"] == payload


def test_get_eval_dataset_404(tmp_path):
    with patch("api.main.DATASETS_DIR", tmp_path):
        resp = client.get("/eval/datasets/missing")
    assert resp.status_code == 404


def test_get_eval_dataset_rejects_traversal(tmp_path):
    with patch("api.main.DATASETS_DIR", tmp_path):
        resp = client.get("/eval/datasets/..%2Fetc%2Fpasswd")
    # FastAPI will URL-decode; either 400 (invalid name) or 404 are acceptable
    assert resp.status_code in (400, 404)


def test_create_eval_dataset(tmp_path):
    with patch("api.main.DATASETS_DIR", tmp_path):
        resp = client.post("/eval/datasets", json={
            "name": "new",
            "samples": [
                {"question": "q1", "ground_truth": "g1", "contexts": ["c"], "answer": "a"},
            ],
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "new"
    assert body["samples"] == 1
    assert (tmp_path / "new.json").exists()


def test_create_eval_dataset_rejects_traversal(tmp_path):
    with patch("api.main.DATASETS_DIR", tmp_path):
        resp = client.post("/eval/datasets", json={
            "name": "../escape",
            "samples": [],
        })
    assert resp.status_code == 400


# ── /eval/runs ────────────────────────────────────────────────────────────────

def test_list_eval_runs(tmp_path):
    (tmp_path / "20260520T000000Z_results.json").write_text(json.dumps({
        "run_id": "20260520T000000Z",
        "dataset": "eval/sample_dataset.json",
        "config": {"top_k": 4},
        "scores": {"hit_rate": 0.5},
    }))
    (tmp_path / "20260521T000000Z_results.json").write_text(json.dumps({
        "run_id": "20260521T000000Z",
        "dataset": "eval/sample_dataset.json",
        "config": {"top_k": 8},
        "scores": {"hit_rate": 0.7},
    }))
    with patch("api.main.EVAL_LOGS_DIR", tmp_path):
        resp = client.get("/eval/runs")
    assert resp.status_code == 200
    body = resp.json()
    # Sorted reverse by filename → newest first
    assert body["runs"][0]["run_id"] == "20260521T000000Z"
    assert body["runs"][0]["scores"]["hit_rate"] == 0.7
    assert len(body["runs"]) == 2


def test_list_eval_runs_skips_invalid_json(tmp_path):
    (tmp_path / "bad_results.json").write_text("not json")
    (tmp_path / "20260521T000000Z_results.json").write_text(json.dumps({
        "run_id": "20260521T000000Z", "scores": {}, "config": {},
    }))
    with patch("api.main.EVAL_LOGS_DIR", tmp_path):
        resp = client.get("/eval/runs")
    assert resp.status_code == 200
    assert len(resp.json()["runs"]) == 1


def test_list_eval_runs_empty_dir():
    resp = client.get("/eval/runs")
    assert resp.status_code == 200
    assert "runs" in resp.json()
