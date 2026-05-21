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

## Latency Tracking

Live eval mode records per-question wall-clock latency (`time.perf_counter()` around the `ask()` call). The eval log stores this as `per_question[i].latency_seconds` and aggregates `mean_latency_seconds` into the top-level `scores` block, so cost / latency tuning sits alongside quality scores.

---

## Remaining Work

### 1. Token-cost tracking in eval log
Latency is tracked; token usage is not. Capture `result.response_metadata["token_usage"]` from `ChatOpenAI` and store as `per_question[i].tokens.{prompt,completion,total}`. Aggregate mean total tokens into `scores.mean_tokens` so cost-per-question is visible in the dashboard.

### 2. Surface latency + runs index in the UI
The Eval Dashboard already reads run logs from disk. Switch it to call `GET /eval/runs` (with a filesystem fallback for local dev), and add latency as a row in the per-question scores table.

### 3. Structured logging
Replace ad-hoc `print()` calls in `eval/runner.py` and the scripts with `logging`. Configure a root logger so the API, UI, scripts, and runner all log consistently. The api.main already imports `logging` for the reranker warning — extend that pattern.

### 4. Save-as-eval-sample UI flow
The Q&A tab returns answer + sources but offers no way to capture the result as an eval sample. Add a "Save as eval sample" button that posts to `POST /eval/datasets` (creating or appending to a chosen dataset). This closes the loop between manual exploration and reproducible eval.

### 5. Per-question RAGAS over static datasets
Live mode populates latency; static mode skips it. Either compute latency in static mode (where `ask()` isn't called, so there's no value to record — make the field absent rather than zero) or document the distinction in the eval log schema. Currently `per_question[i].latency_seconds` is omitted on static runs — verify this is intentional and document.

### 6. pgvector swap (stretch)
Swap Chroma for pgvector via Docker Compose as a drop-in alternative. The `get_vectorstore()` abstraction in `embedder.py` should make this a single-file change. Add a `docker-compose.yml` and a `retriever/pgvector_store.py` that matches the `get_vectorstore()` interface.
