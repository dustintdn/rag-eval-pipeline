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


# ── /query/stream ─────────────────────────────────────────────────────────────

def _stream_chunks(tokens):
    """Yield objects that quack like ChatOpenAI stream chunks."""
    for t in tokens:
        chunk = MagicMock()
        chunk.content = t
        yield chunk


def test_query_stream_yields_tokens_then_sources():
    fake_llm = MagicMock()
    fake_llm.stream.return_value = _stream_chunks(["Hello", " ", "world"])
    fake_docs = [Document(page_content="ctx", metadata={"source_file": "a.txt"})]

    with (
        patch("langchain_openai.ChatOpenAI", return_value=fake_llm),
        patch("retriever.retriever.retrieve", return_value=fake_docs),
    ):
        with client.stream("POST", "/query/stream", json={"question": "What is RAG?"}) as resp:
            assert resp.status_code == 200
            events = [line for line in resp.iter_lines() if line]

    payloads = [json.loads(line.removeprefix("data: ")) for line in events]
    tokens = [p["token"] for p in payloads if "token" in p]
    assert tokens == ["Hello", " ", "world"]
    final = payloads[-1]
    assert final.get("done") is True
    assert final["sources"][0]["content"] == "ctx"


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


# ── Auth ──────────────────────────────────────────────────────────────────────

@patch("api.main.settings.api_token", "secret123")
@patch("api.main.ask", return_value=_mock_qa_result())
def test_query_requires_token_when_configured(mock_ask):
    resp = client.post("/query", json={"question": "What is RAG?"})
    assert resp.status_code == 401


@patch("api.main.settings.api_token", "secret123")
@patch("api.main.ask", return_value=_mock_qa_result())
def test_query_accepts_correct_token(mock_ask):
    resp = client.post(
        "/query",
        json={"question": "What is RAG?"},
        headers={"Authorization": "Bearer secret123"},
    )
    assert resp.status_code == 200


@patch("api.main.settings.api_token", "secret123")
@patch("api.main.ask", return_value=_mock_qa_result())
def test_query_rejects_wrong_token(mock_ask):
    resp = client.post(
        "/query",
        json={"question": "What is RAG?"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


@patch("api.main.settings.api_token", "secret123")
def test_get_endpoints_remain_open_when_auth_configured():
    resp = client.get("/prompts")
    assert resp.status_code == 200


# ── /eval/run/async ───────────────────────────────────────────────────────────

@patch("api.main.run_eval", return_value=("20260521T120000Z", {}))
def test_eval_run_async_returns_job_id(mock_run, tmp_path):
    with patch("api.main.JOBS_DIR", tmp_path):
        resp = client.post("/eval/run/async", json={})
        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["status"] in ("pending", "running", "done")

        # BackgroundTasks runs synchronously inside TestClient; status should be done.
        status = client.get(f"/eval/jobs/{body['job_id']}").json()
        assert status["status"] == "done"
        assert status["run_id"] == "20260521T120000Z"


def test_eval_job_status_404(tmp_path):
    with patch("api.main.JOBS_DIR", tmp_path):
        resp = client.get("/eval/jobs/nonexistent")
    assert resp.status_code == 404


@patch("api.main.run_eval", side_effect=RuntimeError("eval blew up"))
def test_eval_run_async_records_failure(mock_run, tmp_path):
    with patch("api.main.JOBS_DIR", tmp_path):
        resp = client.post("/eval/run/async", json={})
        body = resp.json()
        status = client.get(f"/eval/jobs/{body['job_id']}").json()
        assert status["status"] == "failed"
        assert "eval blew up" in status["error"]


def test_eval_jobs_list(tmp_path):
    (tmp_path / "abc.json").write_text(json.dumps({"status": "done", "run_id": "R1"}))
    (tmp_path / "def.json").write_text(json.dumps({"status": "running"}))
    with patch("api.main.JOBS_DIR", tmp_path):
        resp = client.get("/eval/jobs")
    body = resp.json()
    assert resp.status_code == 200
    job_ids = {j["job_id"] for j in body["jobs"]}
    assert job_ids == {"abc", "def"}


def test_eval_job_state_survives_module_state(tmp_path):
    """Job state lives on disk, so re-reading the same job_id returns the same record."""
    (tmp_path / "persistent.json").write_text(json.dumps({"status": "done", "run_id": "X"}))
    with patch("api.main.JOBS_DIR", tmp_path):
        a = client.get("/eval/jobs/persistent").json()
        b = client.get("/eval/jobs/persistent").json()
    assert a == b
    assert a["run_id"] == "X"


# ── Job TTL cleanup ───────────────────────────────────────────────────────────

def test_prune_old_jobs_removes_terminal_only(tmp_path):
    import os
    import time as _time
    from api.main import _prune_old_jobs

    old = tmp_path / "old_done.json"
    old.write_text(json.dumps({"status": "done"}))
    old_running = tmp_path / "old_running.json"
    old_running.write_text(json.dumps({"status": "running"}))
    recent = tmp_path / "recent_done.json"
    recent.write_text(json.dumps({"status": "done"}))

    # Backdate the "old" files past the TTL
    long_ago = _time.time() - 60 * 86400
    os.utime(old, (long_ago, long_ago))
    os.utime(old_running, (long_ago, long_ago))

    with patch("api.main.JOBS_DIR", tmp_path), patch("api.main.settings.job_ttl_days", 30):
        removed = _prune_old_jobs()

    assert removed == 1
    assert not old.exists()
    assert old_running.exists()  # non-terminal, kept regardless of age
    assert recent.exists()       # within TTL


def test_prune_old_jobs_disabled_when_ttl_zero(tmp_path):
    from api.main import _prune_old_jobs
    (tmp_path / "old.json").write_text(json.dumps({"status": "done"}))
    with patch("api.main.JOBS_DIR", tmp_path), patch("api.main.settings.job_ttl_days", 0):
        removed = _prune_old_jobs()
    assert removed == 0
