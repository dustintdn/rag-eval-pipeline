import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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
    yield


app = FastAPI(title="RAG Eval Pipeline", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def ingest(file: UploadFile = File(...)):
    return await _ingest_one(file)


@app.post("/ingest/batch", dependencies=[Depends(require_token)])
async def ingest_batch(files: list[UploadFile] = File(...)):
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    results = [await _ingest_one(f) for f in files]
    return {"files": results, "total_chunks_added": sum(r["chunks_added"] for r in results)}


@app.post("/query", dependencies=[Depends(require_token)])
def query(req: QueryRequest):
    result = ask(req.question, top_k=req.top_k, prompt_version=req.prompt_version)
    return {
        "answer": result.answer,
        "prompt_version": result.prompt_version,
        "sources": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in result.source_documents
        ],
    }


@app.post("/eval/run", dependencies=[Depends(require_token)])
def eval_run(req: EvalRunRequest | None = None):
    req = req or EvalRunRequest()
    dataset_path = Path(req.dataset) if req.dataset else DEFAULT_DATASET
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found at {dataset_path}")

    overrides: dict = {}
    if req.prompt_version is not None:
        overrides["prompt_version"] = req.prompt_version
    if req.top_k is not None:
        overrides["top_k"] = req.top_k
    if req.enable_reranker is not None:
        overrides["enable_reranker"] = req.enable_reranker

    run_id, _ = run_eval(dataset_path, live=req.live, config_overrides=overrides or None)
    return {"run_id": run_id, "live": req.live, "overrides": overrides}


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
