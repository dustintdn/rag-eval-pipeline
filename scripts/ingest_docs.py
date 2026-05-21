"""CLI: python scripts/ingest_docs.py --source docs/"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.chunker import chunk_documents
from ingest.embedder import embed_and_store
from ingest.loader import load_directory, load_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help="Directory of documents to ingest")
    group.add_argument("--file", help="Single file to ingest")
    args = parser.parse_args()

    if args.source:
        print(f"Loading documents from {args.source}…")
        docs = load_directory(args.source)
    else:
        print(f"Loading {args.file}…")
        docs = load_file(args.file)

    print(f"Loaded {len(docs)} document(s). Chunking…")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks. Embedding and storing…")
    count = embed_and_store(chunks)
    print(f"Done. {count} chunks added to vector store.")


if __name__ == "__main__":
    main()
