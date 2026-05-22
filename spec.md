# Project Spec: RAG Pipeline with Evaluation Framework

## Overview

A production-style document Q&A system with a full evaluation layer. The system ingests documents, answers questions via RAG, and scores itself on both retrieval and generation quality.

---

## Tech Stack

- **Language:** Python 3.12, venv at `.venv/`
- **RAG Framework:** LangChain
- **Vector Store:** Chroma (local, persisted to `./chroma_db`)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **LLM:** OpenAI `gpt-4o-mini`
- **Reranker:** Cohere `rerank-english-v3.0` (optional)
- **Evaluation:** RAGAS 0.2.x
- **API Layer:** FastAPI + uvicorn
- **UI:** Streamlit + Plotly
- **Config:** Pydantic Settings + `.env`
- **Experiment tracking:** Timestamped JSON logs in `eval_logs/`

---

## Project Structure

```
rag-eval-pipeline/
├── .env
├── .env.example
├── requirements.txt
├── README.md
├── spec.md
├── config.py
├── prompts/
│   ├── registry.py
│   ├── v1_cite_sources.json
│   └── v2_concise.json
├── ingest/
│   ├── loader.py
│   ├── chunker.py
│   └── embedder.py
├── retriever/
│   ├── retriever.py
│   └── reranker.py
├── chain/
│   ├── cache.py
│   └── qa_chain.py
├── eval/
│   ├── dataset.py
│   ├── retrieval_metrics.py
│   ├── ragas_eval.py
│   ├── runner.py
│   └── sample_dataset.json
├── api/
│   └── main.py
├── ui/
│   └── app.py
├── scripts/
│   ├── ingest_docs.py
│   └── run_eval.py
├── eval_logs/
├── docs/
└── tests/
    ├── test_chunker.py
    ├── test_retriever.py
    ├── test_eval_metrics.py
    └── test_api.py
```

---

## Configuration

All values overridable via `.env`:

```
OPENAI_API_KEY=
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=rag_docs
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=4
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

PROMPT_VERSION=v1_cite_sources
ENABLE_SEMANTIC_CACHE=false

COHERE_API_KEY=
ENABLE_RERANKER=false
RERANKER_MODEL=rerank-english-v3.0
RERANKER_TOP_N=4
RERANKER_FETCH_K=10
```

---

## Module Responsibilities

### `ingest/`
- `loader.py` — load PDFs, `.txt`, `.md` files into LangChain `Document` objects
- `chunker.py` — split with `RecursiveCharacterTextSplitter`; attach `source_file`, `chunk_index`, `page_number` metadata
- `embedder.py` — embed chunks with OpenAI and upsert into Chroma; expose `get_vectorstore()`

### `prompts/`
- JSON files define versioned prompt templates with `version`, `description`, and `template` fields
- `registry.py` loads a prompt by version ID, returns `(PromptTemplate, metadata)`; exposes `list_versions()`
- Active version set via `PROMPT_VERSION` config; every eval run logs the version used

### `retriever/`
- `retriever.py` — Chroma similarity search, configurable `top_k`
- `reranker.py` — fetches `RERANKER_FETCH_K` candidates from Chroma, reranks via Cohere, returns top `RERANKER_TOP_N`; uses `ContextualCompressionRetriever`

### `chain/`
- `cache.py` — in-process semantic cache; stores `(embedding_vector, QAResult)` pairs; cosine similarity lookup with configurable threshold; module-level singleton
- `qa_chain.py` — builds `RetrievalQA` chain with the correct retriever (plain or reranking) and prompt version; checks and populates the semantic cache on every `ask()` call; returns answer, source documents, and prompt version used

### `eval/`
- `dataset.py` — load/save eval datasets as JSON arrays
- `retrieval_metrics.py` — hit rate and MRR; relevance should be determined by embedding-based cosine similarity, not substring matching
- `ragas_eval.py` — runs RAGAS faithfulness, answer_relevancy, context_precision, context_recall; LLM and embeddings passed explicitly
- `runner.py` — orchestrates a full eval run; config snapshot includes model, chunk settings, top_k, prompt version, reranker state; writes to `eval_logs/{timestamp}_results.json`

### `api/`
- `POST /ingest` — file upload → ingestion pipeline
- `POST /query` — `{"question": "...", "prompt_version": null}` → answer + sources + prompt_version used
- `POST /eval/run` — triggers eval run, returns run ID
- `GET /eval/results/{run_id}` — returns full eval run JSON
- `GET /prompts` — lists available versions and the active one

### `ui/`
- **Ingest tab** — multi-file upload, chunk count on success
- **Q&A tab** — question input, prompt version selector, answer + expandable source chunks
- **Eval Dashboard tab** — run eval, scores table + bar chart, side-by-side run comparison

### `scripts/`
- `ingest_docs.py` — CLI wrapper: `--source <dir>` or `--file <path>`
- `run_eval.py` — CLI wrapper: `--dataset <path>`, `--live` flag to run each question through the live pipeline before scoring

---

## Eval Dataset Format

```json
[
  {
    "question": "What is RAG?",
    "ground_truth": "RAG is retrieval augmented generation...",
    "contexts": ["retrieved chunk 1", "retrieved chunk 2"],
    "answer": "RAG is a technique that improves LLM outputs..."
  }
]
```

`eval/sample_dataset.json` contains 10 QA pairs over the sample docs in `docs/`.

---

## Eval Log Schema

Each run writes `eval_logs/{run_id}_results.json` with:

- `run_id`, `dataset` — identifiers
- `config` — full configuration snapshot (models, chunk settings, top_k, prompt version, reranker state, `live_eval` flag)
- `scores` — mean across questions: `hit_rate`, `mrr`, `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- `per_question[i]` — `question`, `answer`, `ground_truth`, `num_contexts`, and `scores` (per-sample `hit`, `reciprocal_rank`, and all four RAGAS metrics)

Per-question scores enable diagnosing *which* questions fail rather than only seeing aggregate scores.

---

## Ingestion Idempotency

`embed_and_store()` generates stable chunk IDs from `source_file::page_number::chunk_index` (or `source_file::chunk_index` if no page metadata). Chroma upserts on matching IDs, so re-ingesting the same file replaces existing chunks rather than duplicating them.

---

## CLI Overrides

`scripts/run_eval.py` accepts `--prompt-version`, `--top-k`, `--reranker / --no-reranker` flags. Overrides are applied via a `_settings_override` context manager in `eval/runner.py` that mutates the global `settings` singleton for the duration of the run and restores it afterward (even on exception). The applied overrides land in the eval log's `config` block so each run is self-describing.

---

## Eval Datasets and Runs API

| Method | Path | Description |
|---|---|---|
| `GET` | `/eval/datasets` | List `eval/*.json` datasets with sample counts |
| `GET` | `/eval/datasets/{name}` | Return a single dataset's samples |
| `POST` | `/eval/datasets` | Persist a new dataset to `eval/{name}.json` |
| `GET` | `/eval/runs` | List all eval runs (newest first) with their `config` + `scores` |

Dataset name validation rejects `/` and `..` to block path traversal. The runs endpoint silently skips logs whose JSON fails to parse.

---

## Per-Question Observability (Live Mode)

Live eval mode records five per-question fields:

- `latency_seconds` — wall-clock seconds around the `ask()` call (`time.perf_counter`).
- `tokens` — `{prompt, completion, total}` from a `get_openai_callback()` wrapper around `chain.invoke()`. Cache hits skip this field.
- `cost_usd` — `estimate_cost_usd(model, prompt, completion)` against `config.MODEL_PRICING_PER_1K`. Unknown models price as 0.
- `from_cache` — set by `ask()` when the semantic cache short-circuits the LLM call.
- `retrieved` — list of `{source_file, chunk_index, page_number?}` for each retrieved chunk, so the diagnostics view can answer "which chunk is repeatedly winning rank 1?"

Aggregates land in `scores`: `mean_latency_seconds`, `mean_total_tokens`, `total_cost_usd`, `cache_hit_rate`. All are omitted in static mode (`live=False`) because there is no `ask()` call to observe.

---

## Logging

`logger.get_logger(name)` is the single entry point. It configures `logging.basicConfig` once (idempotent across modules), honouring `LOG_LEVEL=DEBUG|INFO|WARN|ERROR` from the environment. API, runner, and scripts all use it. Tests don't depend on log output, so they pass with logging at any level.

---

## Authentication

`API_TOKEN` is an optional config field. When empty (default), all API endpoints are open — preserving local-dev ergonomics. When set, the `require_token` dependency rejects requests to mutating endpoints (`/ingest`, `/ingest/batch`, `/query`, `/eval/run`, `POST /eval/datasets`) unless they carry `Authorization: Bearer <API_TOKEN>`. Read-only endpoints (`/prompts`, `GET /eval/*`) stay open so the UI can call them from a different origin without auth.

---

## Dataset Versioning

`eval/dataset.py::dataset_checksum(samples)` returns a 16-char SHA-256 prefix over the canonical (sort_keys=True) JSON serialisation. Every eval run records `dataset_version` at the top level of the log. The UI comparison view raises a warning when two compared runs reference the same dataset *path* but have different `dataset_version` values — so silent edits to a dataset don't masquerade as pipeline improvements.

---

## Hybrid Retrieval

`ENABLE_HYBRID_RETRIEVAL=true` swaps the dense-only retriever for an `EnsembleRetriever` that combines BM25 (lexical, from `langchain_community.retrievers.BM25Retriever`) with the Chroma dense retriever. `HYBRID_BM25_WEIGHT` (default `0.4`) sets the BM25 share; the dense retriever gets `1 - HYBRID_BM25_WEIGHT`. The BM25 index materialises every chunk from the Chroma collection at retrieval time; for empty collections it falls back to dense-only to avoid the BM25 empty-corpus error.

When both `ENABLE_RERANKER` and `ENABLE_HYBRID_RETRIEVAL` are true, the reranker wraps the hybrid retriever as its base — hybrid produces `RERANKER_FETCH_K` lexical+dense candidates, the reranker scores them, and the top `RERANKER_TOP_N` are returned.

---

## Async Eval Runs

| Method | Path | Description |
|---|---|---|
| `POST` | `/eval/run/async` | Submit an eval; returns `202 + {job_id, status}` |
| `GET` | `/eval/jobs/{job_id}` | Returns `{status: pending\|running\|done\|failed, run_id?, error?}` |
| `GET` | `/eval/jobs` | Lists all jobs (newest first) |

Backed by FastAPI `BackgroundTasks`. Job state is persisted to `eval_logs/.jobs/{job_id}.json` so an API restart preserves history. When a job finishes, the eventual eval `run_id` is populated and `GET /eval/results/{run_id}` returns the full log. The synchronous `POST /eval/run` remains for short / scripted runs.

Job records are pruned on startup: entries in a terminal state (`done` or `failed`) older than `JOB_TTL_DAYS` (default 30) are deleted. Non-terminal jobs are kept regardless of age, so stuck `running` jobs remain visible until the operator intervenes. Set `JOB_TTL_DAYS=0` to disable pruning entirely.

---

## Streaming `/query/stream`

`POST /query/stream` runs retrieval once, then streams the LLM response as Server-Sent Events. Each event payload is a JSON object:

- `{"token": "<piece>"}` — a partial LLM token.
- `{"sources": [...], "prompt_version": "...", "done": true}` — final event with retrieved chunks and the prompt version used.

Implemented by formatting the prompt template against retrieved context, then calling `ChatOpenAI(streaming=True).stream()`. The synchronous `/query` endpoint remains for callers that prefer a single JSON response.

---

## CI Gate: `scripts/eval_compare.py`

`python scripts/eval_compare.py <run_a> <run_b> [--threshold 0.05]` prints a metric-delta table. Exit code 1 when any quality metric (hit_rate, mrr, faithfulness, answer_relevancy, context_precision, context_recall) in run B regresses by more than the threshold; non-quality metrics (latency, cost, tokens) never trip the gate. Warns to stderr when the two runs reference different `dataset_version` values. Suitable for use as a CI gate against a baseline eval log committed to the repo.

---

## Remaining Work

### 1. Streaming Q&A in the UI
The API exposes `/query/stream`, but the Streamlit Q&A tab still calls the synchronous `ask()` and waits for the full answer. Switch the tab to consume SSE from `/query/stream` and render tokens progressively. Requires the UI to talk to the API over HTTP rather than importing `ask` directly.

### 2. Rate limiting on mutating endpoints
Auth gates *who* can call mutating endpoints, not *how often*. `/eval/run` and `/ingest/batch` each call OpenAI in a loop. Add `slowapi` middleware with sensible defaults — `5/minute` on `/eval/run`, `30/minute` on `/query`. Limits should be configurable via env.

### 3. Async UI integration for eval runs
The UI calls `run_eval()` synchronously inside the Streamlit process; long runs block the Streamlit worker. Switch to `POST /eval/run/async` and poll `GET /eval/jobs/{job_id}`.

### 4. Periodic job TTL pruning
Startup-time pruning leaves long-running API processes accumulating job records until next restart. Add an asyncio background task that runs `_prune_old_jobs()` every `JOB_PRUNE_INTERVAL_HOURS` (default 24).

### 5. Streaming token cost accounting
`/query/stream` doesn't run through `get_openai_callback()` (the callback only wraps the chain's `.invoke`, not direct LLM streams). Track tokens via the chunks' `usage_metadata` when available, or fall back to a `tiktoken`-based estimate. Without this, streamed queries don't contribute to the cost picture.

### 6. pgvector swap (stretch)
Swap Chroma for pgvector via Docker Compose as a drop-in alternative. The `get_vectorstore()` abstraction in `embedder.py` should make this a single-file change. Add a `docker-compose.yml` and a `retriever/pgvector_store.py` that matches the `get_vectorstore()` interface.
