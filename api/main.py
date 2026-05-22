import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
from contextlib import asynccontextmanager

import asyncio
import time
import uuid

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from logger import get_logger

logger = get_logger(__name__)

from chain.qa_chain import ask
from config import settings
from eval.dataset import load_dataset, save_dataset
from eval.runner import EVAL_LOGS_DIR, run_eval
from ingest.chunker import chunk_documents
from ingest.embedder import embed_and_store
from ingest.loader import load_file
from prompts.registry import list_versions

DEFAULT_DATASET = Path("eval/sample_dataset.json")
DATASETS_DIR = Path("eval")


async def _periodic_job_pruner() -> None:
    interval_seconds = settings.job_prune_interval_hours * 3600
    if interval_seconds <= 0:
        return
    while True:
        await asyncio.sleep(interval_seconds)
        removed = _prune_old_jobs()
        if removed:
            logger.info("Periodic prune removed %d expired job records", removed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.enable_semantic_cache:
        from chain.cache import enable_semantic_cache
        enable_semantic_cache()
    if settings.enable_reranker and not settings.cohere_api_key:
        logger.warning(
            "ENABLE_RERANKER=true but COHERE_API_KEY is empty — queries will silently "
            "fall back to the plain retriever. Set COHERE_API_KEY or disable the reranker."
        )
    removed = _prune_old_jobs()
    if removed:
        logger.info("Pruned %d expired async job records on startup", removed)
    pruner_task = asyncio.create_task(_periodic_job_pruner())
    try:
        yield
    finally:
        pruner_task.cancel()
        try:
            await pruner_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="RAG Eval Pipeline", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class QueryRequest(BaseModel):
    question: str
    prompt_version: str | None = None
    top_k: int | None = None


class EvalRunRequest(BaseModel):
    live: bool = False
    dataset: str | None = None
    prompt_version: str | None = None
    top_k: int | None = None
    enable_reranker: bool | None = None


class EvalSamplePayload(BaseModel):
    question: str
    ground_truth: str
    contexts: list[str] = []
    answer: str = ""


class EvalDatasetCreateRequest(BaseModel):
    name: str
    samples: list[EvalSamplePayload]


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def require_token(authorization: str | None = Header(default=None)) -> None:
    """Require `Authorization: Bearer <API_TOKEN>` when API_TOKEN is set.

    When API_TOKEN is empty (default), all endpoints are open — preserving
    local-dev ergonomics. When set, mutating endpoints reject requests with
    a missing or mismatched token.
    """
    expected = settings.api_token
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


async def _ingest_one(file: UploadFile) -> dict:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        docs = load_file(tmp_path)
        chunks = chunk_documents(docs)
        count = embed_and_store(chunks)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"filename": file.filename, "chunks_added": count}


@app.post("/ingest", dependencies=[Depends(require_token)])
@limiter.limit(lambda: settings.rate_limit_ingest)
async def ingest(request: Request, file: UploadFile = File(...)):
    return await _ingest_one(file)


@app.post("/ingest/batch", dependencies=[Depends(require_token)])
@limiter.limit(lambda: settings.rate_limit_ingest)
async def ingest_batch(request: Request, files: list[UploadFile] = File(...)):
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    results = [await _ingest_one(f) for f in files]
    return {"files": results, "total_chunks_added": sum(r["chunks_added"] for r in results)}


@app.post("/query", dependencies=[Depends(require_token)])
@limiter.limit(lambda: settings.rate_limit_query)
def query(request: Request, req: QueryRequest):
    result = ask(req.question, top_k=req.top_k, prompt_version=req.prompt_version)
    return {
        "answer": result.answer,
        "prompt_version": result.prompt_version,
        "sources": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in result.source_documents
        ],
    }


@app.post("/query/stream", dependencies=[Depends(require_token)])
@limiter.limit(lambda: settings.rate_limit_query)
def query_stream(request: Request, req: QueryRequest):
    """Stream the LLM response token-by-token as a text/event-stream.

    Retrieval runs once up-front; the prompt is then streamed through
    ChatOpenAI with streaming=True. Each SSE event is a JSON object with
    either {token: str} or {sources: [...], done: true}.
    """
    from langchain_openai import ChatOpenAI

    from prompts.registry import load_prompt
    from retriever.retriever import retrieve

    version = req.prompt_version or settings.prompt_version
    prompt, _ = load_prompt(version)
    docs = retrieve(req.question, top_k=req.top_k)
    formatted = prompt.format(
        context="\n\n".join(d.page_content for d in docs),
        question=req.question,
    )
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
        streaming=True,
    )

    from config import estimate_cost_usd

    def event_stream():
        usage = {"prompt": 0, "completion": 0, "total": 0}
        for chunk in llm.stream(formatted):
            content = getattr(chunk, "content", None)
            if content:
                yield f"data: {json.dumps({'token': content})}\n\n"
            chunk_usage = getattr(chunk, "usage_metadata", None)
            if chunk_usage:
                usage = {
                    "prompt": int(chunk_usage.get("input_tokens", usage["prompt"])),
                    "completion": int(chunk_usage.get("output_tokens", usage["completion"])),
                    "total": int(chunk_usage.get("total_tokens", usage["total"])),
                }
        sources_payload = [
            {"content": d.page_content, "metadata": d.metadata} for d in docs
        ]
        final_payload = {
            "sources": sources_payload,
            "prompt_version": version,
            "done": True,
        }
        if usage["total"] > 0:
            final_payload["tokens"] = usage
            final_payload["cost_usd"] = estimate_cost_usd(
                settings.llm_model, usage["prompt"], usage["completion"]
            )
        yield f"data: {json.dumps(final_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


TERMINAL_JOB_STATES = {"done", "failed"}

# Async-run state. Persisted to disk so restarts don't drop job history.
# Keyed by job_id (assigned at submission); each entry tracks status and the
# eventual eval run_id once the runner completes.
JOBS_DIR = EVAL_LOGS_DIR / ".jobs"


def _prune_old_jobs() -> int:
    """Delete terminal (done/failed) jobs older than JOB_TTL_DAYS. Returns count removed."""
    if not JOBS_DIR.exists() or settings.job_ttl_days <= 0:
        return 0
    cutoff = time.time() - settings.job_ttl_days * 86400
    removed = 0
    for path in JOBS_DIR.glob("*.json"):
        if path.stat().st_mtime >= cutoff:
            continue
        try:
            state = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if state.get("status") in TERMINAL_JOB_STATES:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _read_job(job_id: str) -> dict | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _write_job(job_id: str, state: dict) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    _job_path(job_id).write_text(json.dumps(state))


def _build_overrides(req: EvalRunRequest) -> dict:
    overrides: dict = {}
    if req.prompt_version is not None:
        overrides["prompt_version"] = req.prompt_version
    if req.top_k is not None:
        overrides["top_k"] = req.top_k
    if req.enable_reranker is not None:
        overrides["enable_reranker"] = req.enable_reranker
    return overrides


def _run_eval_job(job_id: str, dataset_path: Path, live: bool, overrides: dict) -> None:
    state = _read_job(job_id) or {}
    state["status"] = "running"
    _write_job(job_id, state)
    try:
        run_id, _ = run_eval(dataset_path, live=live, config_overrides=overrides or None)
        state["run_id"] = run_id
        state["status"] = "done"
    except Exception as exc:  # noqa: BLE001 — surface error to caller via status
        logger.exception("Eval job %s failed", job_id)
        state["status"] = "failed"
        state["error"] = str(exc)
    _write_job(job_id, state)


@app.post("/eval/run", dependencies=[Depends(require_token)])
@limiter.limit(lambda: settings.rate_limit_eval_run)
def eval_run(request: Request, req: EvalRunRequest | None = None):
    req = req or EvalRunRequest()
    dataset_path = Path(req.dataset) if req.dataset else DEFAULT_DATASET
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found at {dataset_path}")

    overrides = _build_overrides(req)
    run_id, _ = run_eval(dataset_path, live=req.live, config_overrides=overrides or None)
    return {"run_id": run_id, "live": req.live, "overrides": overrides}


@app.post("/eval/run/async", status_code=202, dependencies=[Depends(require_token)])
@limiter.limit(lambda: settings.rate_limit_eval_run)
def eval_run_async(request: Request, background_tasks: BackgroundTasks, req: EvalRunRequest | None = None):
    req = req or EvalRunRequest()
    dataset_path = Path(req.dataset) if req.dataset else DEFAULT_DATASET
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found at {dataset_path}")

    job_id = uuid.uuid4().hex
    _write_job(job_id, {"status": "pending", "live": req.live})
    overrides = _build_overrides(req)
    background_tasks.add_task(_run_eval_job, job_id, dataset_path, req.live, overrides)
    return {"job_id": job_id, "status": "pending"}


@app.get("/eval/jobs/{job_id}")
def eval_job_status(job_id: str):
    state = _read_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": job_id, **state}


@app.get("/eval/jobs")
def list_eval_jobs():
    if not JOBS_DIR.exists():
        return {"jobs": []}
    out = []
    for f in sorted(JOBS_DIR.glob("*.json"), reverse=True):
        try:
            out.append({"job_id": f.stem, **json.loads(f.read_text())})
        except json.JSONDecodeError:
            continue
    return {"jobs": out}


@app.get("/eval/results/{run_id}")
def eval_results(run_id: str):
    result_file = EVAL_LOGS_DIR / f"{run_id}_results.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    with open(result_file) as f:
        return JSONResponse(content=json.load(f))


@app.get("/prompts")
def list_prompts():
    return {"versions": list_versions(), "active": settings.prompt_version}


@app.get("/eval/datasets")
def list_eval_datasets():
    if not DATASETS_DIR.exists():
        return {"datasets": []}
    out = []
    for p in sorted(DATASETS_DIR.glob("*.json")):
        try:
            sample_count = len(load_dataset(p))
        except (json.JSONDecodeError, KeyError):
            continue
        out.append({"name": p.stem, "path": str(p), "samples": sample_count})
    return {"datasets": out}


@app.get("/eval/datasets/{name}")
def get_eval_dataset(name: str):
    if "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return {"name": name, "samples": load_dataset(path)}


@app.post("/eval/datasets", dependencies=[Depends(require_token)])
def create_eval_dataset(req: EvalDatasetCreateRequest):
    if "/" in req.name or ".." in req.name or not req.name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    path = DATASETS_DIR / f"{req.name}.json"
    samples = [s.model_dump() for s in req.samples]
    save_dataset(samples, path)
    return {"name": req.name, "path": str(path), "samples": len(samples)}


@app.get("/eval/runs")
def list_eval_runs():
    if not EVAL_LOGS_DIR.exists():
        return {"runs": []}
    runs = []
    for f in sorted(EVAL_LOGS_DIR.glob("*_results.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        runs.append({
            "run_id": data.get("run_id", f.stem.replace("_results", "")),
            "dataset": data.get("dataset"),
            "config": data.get("config", {}),
            "scores": data.get("scores", {}),
        })
    return {"runs": runs}
