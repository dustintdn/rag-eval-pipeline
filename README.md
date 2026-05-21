# RAG Eval Pipeline

A production-style document Q&A system with a full evaluation layer. Ingests documents, answers questions via RAG, and scores itself on both retrieval and generation quality.

## Architecture

```
docs/ ──► loader ──► chunker ──► embedder ──► Chroma
                                                │
user query ──────────────────────────► retriever┘
                                          │
                                     qa_chain (gpt-4o-mini)
                                          │
                                        answer + sources
                                          │
                              eval runner ─► eval_logs/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Quickstart

**Ingest sample docs:**
```bash
python scripts/ingest_docs.py --source docs/
```

**Run the eval pipeline:**
```bash
python scripts/run_eval.py --dataset eval/sample_dataset.json
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
| POST | `/ingest` | Upload a PDF/TXT/MD file for ingestion |
| POST | `/query` | Ask a question, get answer + sources |
| POST | `/eval/run` | Trigger a full eval run |
| GET | `/eval/results/{run_id}` | Fetch eval run results |

## Configuration

All config lives in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `4` | Retrieved chunks per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store path |
| `COLLECTION_NAME` | `rag_docs` | Chroma collection name |

## Sample Eval Results

| Metric | Score |
|---|---|
| Hit Rate | 0.90 |
| MRR | 0.87 |
| Faithfulness | 0.85 |
| Answer Relevancy | 0.88 |
| Context Precision | 0.82 |
| Context Recall | 0.80 |

*Results above are illustrative. Run `scripts/run_eval.py` to generate your own.*

## Project Structure

See `spec.md` for the full specification and definition of done.
