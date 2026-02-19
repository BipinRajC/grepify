#!/usr/bin/env python3
"""Build the vector index: load raw JSON → chunk → embed → upload to Qdrant.

Usage:
    python scripts/index.py
    python scripts/index.py --input data/raw/reddit --collection my_col
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from grepify.chunker import RedditChunker
from grepify.embedder import Embedder
from grepify.scraper import RedditScraper
from grepify.store import QdrantStore


def main():
    ap = argparse.ArgumentParser(description="Build search index")
    ap.add_argument("--input", type=Path, default=Path("data/raw/reddit"))
    ap.add_argument("--collection",
                    default=os.getenv("COLLECTION_NAME", "grepify_finance"))
    ap.add_argument("--model",
                    default=os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2"))
    ap.add_argument("--device",
                    default=os.getenv("EMBEDDING_DEVICE", "cuda"))
    ap.add_argument("--qdrant-url",
                    default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    args = ap.parse_args()

    # 1. Load raw data
    files = sorted(args.input.glob("*.json"))
    if not files:
        print(f"no JSON files in {args.input}")
        return

    threads = []
    for f in files:
        batch = RedditScraper.load(f)
        threads.extend(batch)
        print(f"loaded {len(batch)} threads from {f.name}")
    print(f"total threads: {len(threads)}")

    # 2. Chunk
    chunker = RedditChunker()
    chunks = chunker.chunk_threads(threads)

    tier_counts = {}
    for c in chunks:
        tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
    print(f"chunks: {len(chunks)} ({tier_counts})")

    # 3. Embed
    print(f"embedding model: {args.model} (device={args.device})")
    embedder = Embedder(model_name=args.model, device=args.device)
    vectors = embedder.embed_chunks(chunks)
    print(f"embeddings: {vectors.shape}")

    # 4. Upload
    store = QdrantStore(url=args.qdrant_url)
    store.create_collection(args.collection, dim=embedder.dim)
    store.upload(args.collection, chunks, vectors)
    print(f"\ndone — {store.count(args.collection)} vectors in '{args.collection}'")


if __name__ == "__main__":
    main()
