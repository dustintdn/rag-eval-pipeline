import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from chain.qa_chain import ask
from config import settings
from eval.runner import EVAL_LOGS_DIR, run_eval
from ingest.chunker import chunk_documents
from ingest.embedder import embed_and_store
from ingest.loader import load_file
from prompts.registry import list_versions

DEFAULT_DATASET = Path("eval/sample_dataset.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.enable_semantic_cache:
        from chain.cache import enable_semantic_cache
        enable_semantic_cache()
    yield


app = FastAPI(title="RAG Eval Pipeline", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    prompt_version: str | None = None


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    docs = load_file(tmp_path)
    chunks = chunk_documents(docs)
    count = embed_and_store(chunks)
    Path(tmp_path).unlink(missing_ok=True)

    return {"filename": file.filename, "chunks_added": count}


@app.post("/query")
def query(req: QueryRequest):
    result = ask(req.question, prompt_version=req.prompt_version)
    return {
        "answer": result.answer,
        "prompt_version": result.prompt_version,
        "sources": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in result.source_documents
        ],
    }


@app.post("/eval/run")
def eval_run():
    if not DEFAULT_DATASET.exists():
        raise HTTPException(status_code=404, detail="Default dataset not found at eval/sample_dataset.json")
    run_id, _ = run_eval(DEFAULT_DATASET)
    return {"run_id": run_id}


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
