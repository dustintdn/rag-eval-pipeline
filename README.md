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
```

## Running

**Ingest documents:**
```bash
python scripts/ingest_docs.py --source docs/
# or a single file:
python scripts/ingest_docs.py --file path/to/doc.pdf
```

**Run eval (static — scores pre-baked dataset answers):**
```bash
python scripts/run_eval.py
```

**Run eval (live — retrieves and generates answers in real time):**
```bash
python scripts/run_eval.py --live
```

**Start the API:**
```bash
uvicorn api.main:app --reload
```

**Start the UI:**
```bash
streamlit run ui/app.py
```

**Run tests:**
```bash
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload a PDF, TXT, or MD file |
| `POST` | `/query` | Ask a question — returns answer, sources, and prompt version used |
| `POST` | `/eval/run` | Trigger an eval run against the default dataset |
| `GET` | `/eval/results/{run_id}` | Fetch a past eval run by ID |
| `GET` | `/prompts` | List available prompt versions |

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

## Architecture

```
docs/ ──► loader ──► chunker ──► embedder ──► Chroma
                                                 │
user query ──────────────────────────► retriever ┘
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
- **Live eval mode** — `--live` runs each eval question through the real retriever and chain before scoring. This is what makes scores meaningful when tuning `top_k`, prompt version, or the reranker.
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
├── prompts/               # Versioned prompt templates + registry
├── ingest/                # Document loading, chunking, embedding
├── retriever/             # Chroma retriever + Cohere reranker
├── chain/                 # QA chain, semantic cache
├── eval/                  # Dataset, retrieval metrics, RAGAS, runner
├── api/                   # FastAPI app
├── ui/                    # Streamlit app
├── scripts/               # CLI entry points
└── tests/                 # Unit + API tests (27 total)
```
