# Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances large language model (LLM) outputs by grounding them in external, up-to-date knowledge retrieved at inference time. Instead of relying solely on information baked into model weights during training, RAG pulls relevant documents from a vector store and injects them into the prompt.

## Core Components

### 1. Document Ingestion
Raw documents (PDFs, text files, web pages) are loaded, split into overlapping chunks, and embedded into a dense vector representation. These vectors are stored in a vector database such as Chroma, Pinecone, or pgvector.

### 2. Retrieval
At query time, the user's question is embedded using the same embedding model. A similarity search (typically cosine similarity or dot product) retrieves the top-k most relevant chunks from the vector store.

### 3. Generation
The retrieved chunks are concatenated with the original question and passed to an LLM as context. The model generates an answer grounded in that context rather than hallucinating from parametric memory.

## Why RAG?

- **Reduced hallucination** — the model must justify claims using provided context.
- **Up-to-date knowledge** — the knowledge base can be updated without retraining.
- **Source attribution** — retrieved chunks can be cited, improving trust and verifiability.
- **Cost efficiency** — smaller models perform better when given good context.

## Chunking Strategy

Chunk size and overlap are critical hyperparameters. Smaller chunks (256–512 tokens) improve retrieval precision but may lose necessary surrounding context. Larger chunks increase recall but dilute relevance. Overlap (typically 10–20% of chunk size) prevents key information from being split across chunk boundaries.

## Evaluation Dimensions

RAG systems are evaluated across two axes:

- **Retrieval quality**: Did the right chunks come back? Measured by Hit Rate, MRR, NDCG.
- **Generation quality**: Is the answer faithful and relevant? Measured by RAGAS metrics: faithfulness, answer relevancy, context precision, and context recall.
