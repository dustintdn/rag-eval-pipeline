# Vector Stores for RAG

## Overview

A vector store (also called a vector database) indexes high-dimensional embedding vectors and supports efficient approximate nearest-neighbor (ANN) search. In RAG pipelines, the vector store holds the embedded document chunks and serves retrieval queries at inference time.

## Popular Options

### Chroma
Chroma is an open-source, local-first vector store built for AI applications. It requires no external infrastructure — data is persisted to disk and loaded in-process. Ideal for prototyping and single-server deployments.

Key properties:
- SQLite-backed metadata storage
- HNSW index for ANN search
- Python and JavaScript clients
- Supports persistent and in-memory modes

### Pinecone
A managed, cloud-native vector database. Scales to billions of vectors with low-latency search. Requires an API key and incurs per-query costs. Best for production workloads where operational overhead must be minimized.

### pgvector
An open-source PostgreSQL extension that adds vector similarity search. Allows teams already running Postgres to add embedding search without a separate service. Supports exact and approximate search. Integrates naturally with existing SQL queries and access control.

### Weaviate
A cloud-native vector search engine with built-in vectorization modules, multi-tenancy, and hybrid search (vector + keyword BM25). Suitable for complex retrieval use cases.

### Qdrant
A Rust-based vector store with a focus on filtering and payload search. Offers both cloud and self-hosted options with strong consistency guarantees.

## Choosing a Vector Store

| Factor | Chroma | pgvector | Pinecone |
|---|---|---|---|
| Setup complexity | Minimal | Low (needs Postgres) | None (managed) |
| Scale | Small-medium | Medium-large | Very large |
| Cost | Free | Infrastructure cost | Per-query pricing |
| Operational overhead | None | Low | None |

## HNSW Index

Hierarchical Navigable Small World (HNSW) is the dominant ANN algorithm used by modern vector stores. It builds a layered graph where each node connects to its nearest neighbors. Search traverses from a coarse top layer to a fine bottom layer, achieving sub-linear query time at the cost of memory.

Key parameters:
- `M` — number of connections per node (higher = more accurate, more memory)
- `ef_construction` — search width during index build (higher = better quality, slower build)
- `ef` — search width at query time (higher = better recall, slower query)
