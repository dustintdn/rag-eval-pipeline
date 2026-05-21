# Project Spec: RAG Pipeline with Evaluation Framework

## Overview

Build a production-style document Q&A system with a full evaluation layer. The system should ingest documents, answer questions via RAG, and score itself on both retrieval and generation quality. The goal is a clean, modular codebase that demonstrates senior AI engineering practices.

---

## Tech Stack

- **Language:** Python 3.11+
- **RAG Framework:** LangChain
- **Vector Store:** Chroma (local, no Docker needed)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **LLM:** OpenAI `gpt-4o-mini` (cost-efficient, swap via config)
- **Evaluation:** RAGAS
- **API Layer:** FastAPI
- **UI:** Streamlit
- **Config:** Pydantic Settings + `.env`
- **Experiment tracking:** JSON logs (simple, no overhead)

---

## Project Structure

```
rag-eval-pipeline/
├── .env.example
├── requirements.txt
├── README.md
├── config.py                  # Pydantic settings for all env vars + model config
├── ingest/
│   ├── __init__.py
│   ├── loader.py              # Load PDFs, .txt, .md via LangChain document loaders
│   ├── chunker.py             # Recursive text splitter, configurable chunk_size/overlap
│   └── embedder.py            # Embed + upsert into Chroma
├── retriever/
│   ├── __init__.py
│   └── retriever.py           # Chroma retriever, configurable top_k and similarity threshold
├── chain/
│   ├── __init__.py
│   └── qa_chain.py            # LangChain RetrievalQA chain, returns answer + source docs
├── eval/
│   ├── __init__.py
│   ├── dataset.py             # Load/save eval datasets as JSON [{question, ground_truth, contexts, answer}]
│   ├── retrieval_metrics.py   # Hit rate and MRR computed from retrieved doc chunks
│   ├── ragas_eval.py          # RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
│   └── runner.py              # Orchestrates a full eval run, writes results to eval_logs/
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI: POST /ingest, POST /query, POST /eval/run, GET /eval/results
├── ui/
│   └── app.py                 # Streamlit: file upload, Q&A chat, eval results dashboard
├── scripts/
│   ├── ingest_docs.py         # CLI: python scripts/ingest_docs.py --source docs/
│   └── run_eval.py            # CLI: python scripts/run_eval.py --dataset eval/sample_dataset.json
├── eval_logs/                 # Timestamped JSON eval run outputs
├── docs/                      # Sample documents to ingest (include 3-5 .pdf or .md files)
└── tests/
    ├── test_chunker.py
    ├── test_retriever.py
    └── test_eval_metrics.py
```

---

## Core Features to Build

### 1. Ingestion Pipeline

- Accept PDF, `.txt`, and `.md` files
- Chunk with `RecursiveCharacterTextSplitter` — expose `chunk_size` and `chunk_overlap` as config values
- Embed with OpenAI and persist to a local Chroma collection
- Metadata per chunk: `source_file`, `chunk_index`, `page_number` (if PDF)

### 2. Retrieval + QA Chain

- Chroma similarity search with configurable `top_k`
- `RetrievalQA` chain with a custom prompt template that instructs the model to cite sources
- Return both the answer and the retrieved source chunks on every query

### 3. Retrieval Evaluation Metrics

Implement these from scratch in `retrieval_metrics.py` — do not use a library:

- **Hit Rate:** % of questions where at least one retrieved chunk contains the ground truth answer
- **MRR (Mean Reciprocal Rank):** average of 1/rank of the first relevant chunk

### 4. RAGAS Generation Evaluation

In `ragas_eval.py`, run the following RAGAS metrics against a dataset:

- `faithfulness` — is the answer grounded in the retrieved context?
- `answer_relevancy` — does the answer address the question?
- `context_precision` — are the retrieved chunks actually relevant?
- `context_recall` — does the context cover what's needed to answer?

### 5. Eval Dataset Format

All eval datasets as JSON arrays:

```json
[
  {
    "question": "What is AirOps used for?",
    "ground_truth": "AirOps is a content engineering platform...",
    "contexts": ["retrieved chunk 1", "retrieved chunk 2"],
    "answer": "AirOps helps brands get found via AI-driven platforms..."
  }
]
```

Include a `sample_dataset.json` with 10 hand-crafted QA pairs over your sample docs.

### 6. Eval Runner + Logging

- `runner.py` runs the full pipeline: loads dataset → runs retrieval metrics → runs RAGAS → writes output
- Save each run to `eval_logs/{timestamp}_results.json` with: all scores, config snapshot (model, chunk_size, top_k), and per-question breakdowns

### 7. FastAPI Layer

- `POST /ingest` — accepts file upload, runs ingestion pipeline
- `POST /query` — accepts `{"question": "..."}`, returns answer + sources
- `POST /eval/run` — triggers eval run against default dataset, returns run ID
- `GET /eval/results/{run_id}` — returns the eval run JSON

### 8. Streamlit UI

Three tabs:

- **Ingest** — drag-and-drop file upload, shows chunk count on success
- **Q&A** — chat interface showing answer + expandable source chunks
- **Eval Dashboard** — run eval, display scores as a table + bar charts per metric, allow comparison of two past runs side-by-side

---

## Configuration (`config.py`)

All values should be overridable via `.env`:

```
OPENAI_API_KEY=
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=rag_docs
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=4
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

---

## Stretch Goals (do these after the core is working)

1. **Prompt versioning** — store prompt templates with a version ID, log which version was used in each eval run so you can compare prompts A/B style
2. **Semantic caching** — use `langchain_community.cache.InMemorySemanticCache` to skip redundant LLM calls on near-identical questions
3. **Reranker** — add a Cohere rerank step between retrieval and generation, measure its impact on RAGAS scores
4. **pgvector swap** — swap Chroma for pgvector via Docker Compose as a drop-in, demonstrating the abstraction holds

---

## Definition of Done

- [ ] `scripts/ingest_docs.py` runs end-to-end on sample docs
- [ ] `scripts/run_eval.py` produces a complete JSON report with all 6 metrics
- [ ] FastAPI server starts with `uvicorn api.main:app`
- [ ] Streamlit UI runs with `streamlit run ui/app.py`
- [ ] All three `tests/` pass with `pytest`
- [ ] README documents setup, architecture decisions, and a sample eval results table
