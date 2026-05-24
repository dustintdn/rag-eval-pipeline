# RAG Eval Pipeline

A production-style document Q&A system with an evaluation layer wired in from day one. It ingests documents, answers questions via retrieval-augmented generation, and scores itself on retrieval and generation quality — so every configuration change produces a comparable, persisted eval run.

## What this project is

Most RAG tutorials stop at "the model answers a question." That leaves you guessing whether a longer chunk size, a different prompt, or a Cohere reranker actually helped. This repo bundles three things into one workflow:

1. **An ingestion + retrieval + generation pipeline** — PDFs / Markdown / TXT in, cited answers out, with optional hybrid (BM25 + dense) retrieval and a Cohere reranker.
2. **An evaluation framework** — embedding-based retrieval metrics (hit rate, MRR) plus RAGAS metrics (faithfulness, answer relevancy, context precision, context recall), recorded per question and aggregated per run.
3. **A serving layer** — FastAPI for programmatic access (sync, streaming, async-job eval runs) and a Streamlit UI for browsing answers, retrieval diagnostics, and side-by-side run comparisons.

Every eval run captures a snapshot of the configuration that produced it — model, chunk settings, top-k, prompt version, reranker / hybrid flags, dataset checksum — so the JSON log is enough to reproduce or compare against later. A CI gate (`scripts/eval_compare.py`) fails the build if a run regresses against a committed baseline.

### Why it's built this way

- **Evaluation is the product feature.** It is the only way to tell whether a knob made things better or worse. The repo treats it as a first-class concern, not an afterthought.
- **Live eval mode is the honest one.** Static eval scores pre-baked dataset answers; live eval re-runs each question through the actual retriever and LLM before scoring. The latter is the only setting that surfaces regressions from a prompt change or a retriever swap.
- **Self-describing runs.** A run log embeds its own config and dataset checksum, so six months later you can still tell what was being tested.
- **Production rough edges.** Bearer-token auth, per-token rate limiting, async background jobs with disk persistence and TTL pruning, SSE streaming with token accounting, structured logging, Docker Compose.

## Architecture

```
docs/ ──► loader ──► chunker ──► embedder ──► Chroma
                                                 │
user query ──────────────────────► BM25 + dense ─┘  ← hybrid, optional
                                           │
                                    [Cohere reranker]  ← optional
                                           │
                                  prompt template (versioned)
                                           │
                                      ChatOpenAI
                                           │
                                    answer + sources
                                           │
                              eval runner ──► eval_logs/
```

**Key design decisions:**

- **Prompt versioning.** Templates live as JSON in `prompts/` and are loaded by version ID. Every eval run records the version it used so A/B comparisons are deterministic.
- **Semantic cache.** Optional in-process cosine-similarity cache wraps `ask()`. Same (or near-identical) question twice → skips the LLM entirely. Opt in via `ENABLE_SEMANTIC_CACHE=true`.
- **Hybrid retrieval.** BM25 sparse scores fused with dense embeddings via configurable weight. Opt in via `ENABLE_HYBRID_RETRIEVAL=true`. When both reranker and hybrid are on, the reranker scores hybrid's candidates.
- **Embedding-based retrieval metrics.** Hit rate and MRR use cosine similarity (threshold 0.75) between ground-truth and chunk embeddings, not substring matching, so paraphrased contexts still count.
- **Async eval jobs.** `POST /eval/run/async` returns a `job_id` immediately and persists job state to disk under `eval_logs/.jobs/`. Terminal jobs are pruned after `JOB_TTL_DAYS`.
- **Streaming.** `/query/stream` emits Server-Sent Events token-by-token, then a final event with sources, prompt version, and token usage (exact when the model reports it, `tiktoken`-estimated otherwise).
- **Idempotent ingestion.** Chunk IDs are deterministic (`source_file::page_number::chunk_index`), so re-ingesting a file upserts rather than duplicates.

## Quickstart

### Prerequisites

- An OpenAI API key (required).
- Optional: a Cohere API key if you want to use the reranker.
- Either Docker (for the compose path) or Python 3.12 (for the manual path).

### 1. Configure

```bash
git clone <repo>
cd rag-eval-pipeline
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY.
# Optional: COHERE_API_KEY, API_TOKEN (bearer auth), feature flags.
```

### 2. Run with Docker Compose (recommended)

```bash
docker compose up
# API at http://localhost:8000  (docs at /docs)
# UI  at http://localhost:8501
```

Compose mounts `chroma_db/`, `eval_logs/`, `eval/`, and `docs/` so state survives rebuilds.

### 2. Run manually

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ingest some documents (skip if you already ingested via the UI or API)
python scripts/ingest_docs.py --source docs/
# ...or a single file:
python scripts/ingest_docs.py --file path/to/doc.pdf

# Start the API
uvicorn api.main:app --reload

# Start the UI (in another terminal)
streamlit run ui/app.py
```

### 3. Try it

- **UI** — open `http://localhost:8501`. The **Ingest** tab uploads documents, **Q&A** asks questions with a chosen prompt version and shows retrieved chunks, **Eval Dashboard** runs evals and compares runs side-by-side.
- **API** — `curl http://localhost:8000/docs` for the interactive OpenAPI spec.
- **CLI** — `python scripts/run_eval.py --live` to score the sample dataset end-to-end.

## Using the system

### Ingesting documents

Supported file types: `.pdf`, `.txt`, `.md`.

```bash
# CLI
python scripts/ingest_docs.py --source docs/
python scripts/ingest_docs.py --file path/to/doc.pdf

# API (single file)
curl -X POST http://localhost:8000/ingest \
  -F "file=@docs/sample.pdf"

# API (batch)
curl -X POST http://localhost:8000/ingest/batch \
  -F "files=@a.pdf" -F "files=@b.md"
```

Re-ingesting the same file replaces existing chunks rather than duplicating them.

### Asking questions

```bash
# JSON response
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'

# Streaming SSE — token events followed by a final {sources, done, tokens, cost_usd}
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

Both endpoints accept optional `prompt_version` and `top_k` overrides per request.

### Running evals

**Static mode** scores the pre-baked `answer`/`contexts` fields in the dataset — useful for sanity checks but doesn't exercise your pipeline.

**Live mode** runs each question through the real retriever and LLM, then scores. This is the mode that tells you whether a config change moved the needle.

```bash
# CLI — static
python scripts/run_eval.py

# CLI — live, with per-run overrides
python scripts/run_eval.py --live --prompt-version v2_concise --top-k 6 --reranker

# API — synchronous (short runs)
curl -X POST http://localhost:8000/eval/run \
  -H "Content-Type: application/json" \
  -d '{"live": true, "prompt_version": "v2_concise"}'

# API — asynchronous (long runs)
curl -X POST http://localhost:8000/eval/run/async \
  -H "Content-Type: application/json" \
  -d '{"live": true}'
# → {"job_id": "...", "status": "pending"}

curl http://localhost:8000/eval/jobs/<job_id>
# → {"status": "done", "run_id": "20260522T..."}

curl http://localhost:8000/eval/results/<run_id>
```

Each run writes `eval_logs/{run_id}_results.json` containing the config snapshot, aggregate scores, and per-question scores.

### Comparing runs (CI gate)

```bash
python scripts/eval_compare.py <run_a_id> <run_b_id> --threshold 0.05
```

Prints a metric-delta table. Exits non-zero when a quality metric in run B regresses by more than the threshold. Latency / cost / token metrics never trip the gate. Wire this into CI against a baseline log committed to the repo and you get automatic protection against silent quality regressions.

## API reference

Write endpoints (`POST`) require `Authorization: Bearer <API_TOKEN>` when `API_TOKEN` is set. Read endpoints stay open so the UI can call them cross-origin.

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload a single PDF, TXT, or MD file |
| `POST` | `/ingest/batch` | Upload multiple files in one request |
| `POST` | `/query` | Ask a question — returns answer, sources, prompt version |
| `POST` | `/query/stream` | Streaming SSE variant — `{token}` events then `{sources, done, tokens?, cost_usd?}` |
| `POST` | `/eval/run` | Run eval synchronously; returns `run_id` when complete |
| `POST` | `/eval/run/async` | Submit an eval job in the background; returns `job_id` (202) |
| `GET` | `/eval/jobs` | List all async eval jobs (newest first) |
| `GET` | `/eval/jobs/{job_id}` | Poll status of an async eval job |
| `GET` | `/eval/results/{run_id}` | Fetch a completed eval run by ID |
| `GET` | `/eval/runs` | List all completed eval runs with config + scores |
| `GET` | `/eval/datasets` | List available eval datasets |
| `GET` | `/eval/datasets/{name}` | Fetch samples from a named dataset |
| `POST` | `/eval/datasets` | Create a new eval dataset |
| `GET` | `/prompts` | List available prompt versions and the active version |

Interactive OpenAPI docs live at `http://localhost:8000/docs`.

## Configuration

All values are read from `.env` via Pydantic Settings.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store path |
| `COLLECTION_NAME` | `rag_docs` | Chroma collection name |
| `PROMPT_VERSION` | `v1_cite_sources` | Active prompt template |
| `ENABLE_SEMANTIC_CACHE` | `false` | In-process cosine-similarity cache |
| `COHERE_API_KEY` | — | Required when reranker is on |
| `ENABLE_RERANKER` | `false` | Cohere rerank step |
| `RERANKER_MODEL` | `rerank-english-v3.0` | Cohere reranker model |
| `RERANKER_FETCH_K` | `10` | Candidates fetched before reranking |
| `RERANKER_TOP_N` | `4` | Chunks kept after reranking |
| `ENABLE_HYBRID_RETRIEVAL` | `false` | BM25 + dense hybrid retrieval |
| `HYBRID_BM25_WEIGHT` | `0.4` | BM25 weight in hybrid mode (dense gets `1 - weight`) |
| `API_TOKEN` | — | Bearer token for write endpoints; unset = no auth |
| `JOB_TTL_DAYS` | `30` | Days before terminal async jobs are pruned (0 disables) |
| `JOB_PRUNE_INTERVAL_HOURS` | `24` | How often the pruning job runs |
| `RATE_LIMIT_QUERY` | `30/minute` | Rate limit for `/query` and `/query/stream` |
| `RATE_LIMIT_EVAL_RUN` | `5/minute` | Rate limit for eval-run endpoints |
| `RATE_LIMIT_INGEST` | `10/minute` | Rate limit for ingest endpoints |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARN`, `ERROR` |

### Authentication

`API_TOKEN` is empty by default for local-dev ergonomics. Set it in production and mutating endpoints (`/ingest`, `/ingest/batch`, `/query`, `/query/stream`, `/eval/run`, `/eval/run/async`, `POST /eval/datasets`) will require `Authorization: Bearer <API_TOKEN>`. Read-only endpoints stay open.

### Rate limiting

Defaults apply per IP. When a request carries a bearer token, the limiter keys by token instead — so callers behind a NAT or load balancer don't share a bucket, and distinct bearers from the same IP stay independent. Over-limit responses are `429`.

## Eval output

Each run writes `eval_logs/{run_id}_results.json`:

- `run_id`, `dataset`, `dataset_version` — identifiers (`dataset_version` is a 16-char SHA-256 prefix of the canonical sample JSON, so silent dataset edits are detected).
- `config` — full snapshot: models, chunk settings, top-k, prompt version, reranker / hybrid flags, `live_eval`.
- `scores` — means across questions: `hit_rate`, `mrr`, `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`. Live mode adds `mean_latency_seconds`, `mean_total_tokens`, `total_cost_usd`, `cache_hit_rate`.
- `per_question[i]` — `question`, `answer`, `ground_truth`, `num_contexts`, optional `latency_seconds`, `tokens`, `cost_usd`, `from_cache`, `retrieved` (which chunks were returned, ranked), and per-sample scores.

Per-question scores are what makes debugging tractable: you can tell *which* questions fail rather than only seeing aggregate scores move.

## Sample results

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

## Project layout

```
rag-eval-pipeline/
├── config.py              # Pydantic Settings + pricing table
├── logger.py              # Structured logging setup
├── Dockerfile
├── docker-compose.yml
├── prompts/               # Versioned prompt templates + registry
├── ingest/                # Loader, chunker, embedder
├── retriever/             # Dense (Chroma), BM25 hybrid, Cohere reranker
├── chain/                 # QA chain, semantic cache, token accounting
├── eval/                  # Datasets, retrieval metrics, RAGAS, runner
├── api/                   # FastAPI app
├── ui/                    # Streamlit app
├── scripts/               # CLI entry points (ingest, run_eval, eval_compare)
├── tests/                 # Unit + API tests
├── docs/                  # Sample documents
└── eval_logs/             # Per-run JSON logs (+ .jobs/ for async state)
```

## Development

```bash
# Run the test suite
pytest tests/ -v

# Tail logs
LOG_LEVEL=DEBUG uvicorn api.main:app --reload
```

Useful entry points:

- `chain/qa_chain.py::ask()` — the single function the API and live eval mode both call.
- `eval/runner.py::run_eval()` — orchestrates a full eval run; accepts a `config_overrides` dict so the CLI flags, API request body, and async jobs all share one path.
- `prompts/registry.py` — drop a new `vN_*.json` file in `prompts/` and it is immediately selectable via `PROMPT_VERSION` or `prompt_version` overrides.
