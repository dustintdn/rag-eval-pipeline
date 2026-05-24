# RAG Eval Pipeline

A production-style document Q&A system with a full evaluation layer. Ingests documents, answers questions via RAG, and measures itself on retrieval and generation quality across every configuration change.

## Setup

```bash
git clone <repo>
cd rag-eval-pipeline

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in OPENAI_API_KEY (required)
# Fill in COHERE_API_KEY if using the reranker
# Fill in API_TOKEN to require bearer auth on write endpoints
```

## Running

**With Docker Compose (recommended):**
```bash
docker compose up
# API at http://localhost:8000
# UI  at http://localhost:8501
```

**Manually:**

```bash
# Ingest documents
python scripts/ingest_docs.py --source docs/
python scripts/ingest_docs.py --file path/to/doc.pdf   # single file

# Run eval (static — scores pre-baked dataset answers)
python scripts/run_eval.py

# Run eval (live — retrieves and generates answers in real time)
python scripts/run_eval.py --live

# Start the API
uvicorn api.main:app --reload

# Start the UI
streamlit run ui/app.py

# Run tests
pytest tests/ -v
```

## API Endpoints

Write endpoints (`POST`) require a `Bearer <API_TOKEN>` header when `API_TOKEN` is set.

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload a single PDF, TXT, or MD file |
| `POST` | `/ingest/batch` | Upload multiple files in one request |
| `POST` | `/query` | Ask a question — returns answer, sources, and prompt version |
| `POST` | `/query/stream` | Streaming SSE variant — emits `{token}` events then a final `{sources, done}` event |
| `POST` | `/eval/run` | Run eval synchronously; returns `run_id` when complete |
| `POST` | `/eval/run/async` | Submit an eval job in the background; returns `job_id` (202) |
| `GET` | `/eval/jobs` | List all async eval jobs |
| `GET` | `/eval/jobs/{job_id}` | Poll status of an async eval job |
| `GET` | `/eval/results/{run_id}` | Fetch a completed eval run by ID |
| `GET` | `/eval/runs` | List all completed eval runs with scores |
| `GET` | `/eval/datasets` | List available eval datasets |
| `GET` | `/eval/datasets/{name}` | Fetch samples from a named dataset |
| `POST` | `/eval/datasets` | Create a new eval dataset |
| `GET` | `/prompts` | List available prompt versions and active version |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store path |
| `PROMPT_VERSION` | `v1_cite_sources` | Active prompt template |
| `ENABLE_SEMANTIC_CACHE` | `false` | In-process cosine similarity cache |
| `COHERE_API_KEY` | — | Required when reranker is on |
| `ENABLE_RERANKER` | `false` | Cohere rerank step |
| `RERANKER_FETCH_K` | `10` | Candidates fetched before reranking |
| `RERANKER_TOP_N` | `4` | Chunks kept after reranking |
| `ENABLE_HYBRID_RETRIEVAL` | `false` | BM25 + dense hybrid retrieval |
| `HYBRID_BM25_WEIGHT` | `0.4` | BM25 score weight in hybrid mode |
| `API_TOKEN` | — | Bearer token for write endpoints; unset = no auth |
| `JOB_TTL_DAYS` | `30` | Days before terminal async jobs are pruned |
| `JOB_PRUNE_INTERVAL_HOURS` | `24` | How often the pruning job runs |
| `RATE_LIMIT_QUERY` | `30/minute` | Rate limit for `/query` and `/query/stream` |
| `RATE_LIMIT_EVAL_RUN` | `5/minute` | Rate limit for eval run endpoints |
| `RATE_LIMIT_INGEST` | `10/minute` | Rate limit for ingest endpoints |

Rate limits are applied per IP. If `API_TOKEN` is set, limits are applied per token instead.

## Architecture

```
docs/ ──► loader ──► chunker ──► embedder ──► Chroma
                                                 │
user query ──────────────────────► BM25 + dense ─┘  ← hybrid, optional
                                           │
                                    [Cohere reranker]  ← optional
                                           │
                                      qa_chain (LLM)
                                           │
                                    answer + sources
                                           │
                              eval runner ──► eval_logs/
```

**Key design decisions:**

- **Prompt versioning** — templates live as JSON files in `prompts/`. Every eval run logs the version used, enabling A/B comparison across runs.
- **Semantic cache** — in-process cosine similarity cache wraps `ask()`. Same question asked twice skips the LLM entirely. Opt-in via `ENABLE_SEMANTIC_CACHE=true`.
- **Hybrid retrieval** — BM25 sparse scores fused with dense embeddings via configurable weight. Opt-in via `ENABLE_HYBRID_RETRIEVAL=true`.
- **Live eval mode** — `--live` runs each eval question through the real retriever and chain before scoring. This is what makes scores meaningful when tuning `top_k`, prompt version, or the reranker.
- **Async eval jobs** — `POST /eval/run/async` submits an eval as a background task and returns immediately. Poll `GET /eval/jobs/{job_id}` for status. Completed jobs are persisted to disk and pruned after `JOB_TTL_DAYS`.
- **Streaming** — `/query/stream` emits tokens via Server-Sent Events as they arrive from the LLM, with a final event containing sources and token cost.
- **Embedding-based retrieval metrics** — hit rate and MRR use cosine similarity between ground truth and chunk embeddings (threshold 0.75), not substring matching.

## Eval Results

Scores on `eval/sample_dataset.json` (10 questions over 3 sample documents), live mode.

| Metric | Baseline | + Cohere Reranker |
|---|---|---|
| Hit Rate | 0.30 | 0.30 |
| MRR | 0.30 | 0.30 |
| Faithfulness | 0.77 | **0.90** |
| Answer Relevancy | 0.81 | **0.92** |
| Context Precision | 0.83 | **0.90** |
| Context Recall | 0.90 | **1.00** |

Config: `gpt-4o-mini`, `text-embedding-3-small`, `chunk_size=512`, `top_k=4`, `prompt=v1_cite_sources`. Reranker fetches 10, keeps 4 via `rerank-english-v3.0`.

## Project Structure

```
rag-eval-pipeline/
├── config.py              # All settings via Pydantic + .env
├── logger.py              # Structured logging setup
├── Dockerfile
├── docker-compose.yml
├── prompts/               # Versioned prompt templates + registry
├── ingest/                # Document loading, chunking, embedding
├── retriever/             # Chroma retriever, BM25, Cohere reranker
├── chain/                 # QA chain, semantic cache, token accounting
├── eval/                  # Dataset management, retrieval metrics, RAGAS, runner
├── api/                   # FastAPI app
├── ui/                    # Streamlit app
├── scripts/               # CLI entry points
└── tests/                 # Unit + API tests
```
