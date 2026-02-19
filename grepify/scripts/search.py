#!/usr/bin/env python3
"""Search the index with full retrieval pipeline.

Usage:
    python scripts/search.py "best mutual fund for SIP"
    python scripts/search.py "credit card fraud" --no-rerank
    python scripts/search.py "UTI nifty index fund" --mode bm25
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from grepify.embedder import Embedder
from grepify.retriever import Retriever
from grepify.store import QdrantStore


def main():
    ap = argparse.ArgumentParser(description="Search the index")
    ap.add_argument("query", help="Search query")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--mode", default="hybrid",
                    choices=["hybrid", "dense", "bm25"],
                    help="Retrieval mode")
    ap.add_argument("--no-rerank", action="store_true",
                    help="Skip cross-encoder reranking")
    ap.add_argument("--mmr-lambda", type=float, default=0.7,
                    help="MMR lambda (0=max diversity, 1=max relevance)")
    ap.add_argument("--collection",
                    default=os.getenv("COLLECTION_NAME", "grepify_finance"))
    ap.add_argument("--model",
                    default=os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2"))
    ap.add_argument("--device",
                    default=os.getenv("EMBEDDING_DEVICE", "cuda"))
    ap.add_argument("--qdrant-url",
                    default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    args = ap.parse_args()

    embedder = Embedder(model_name=args.model, device=args.device)
    store = QdrantStore(url=args.qdrant_url)
    retriever = Retriever(store, embedder, collection=args.collection)

    if args.mode == "hybrid":
        results = retriever.retrieve(
            args.query,
            top_k=args.limit,
            mmr_lambda=args.mmr_lambda,
            rerank=not args.no_rerank,
        )
    elif args.mode == "dense":
        vec = embedder.embed_query(args.query)
        results = store.search(args.collection, vec, limit=args.limit)
    elif args.mode == "bm25":
        retriever._ensure_bm25()
        results = retriever._bm25_search(args.query, limit=args.limit)

    # Display
    print(f"\nQuery: {args.query}")
    print(f"Mode: {args.mode} | Rerank: {not args.no_rerank} | MMR λ: {args.mmr_lambda}")
    print(f"Results: {len(results)}")
    print("=" * 80)

    for i, r in enumerate(results, 1):
        tier = r.get("tier", "?")
        sub = r.get("subreddit", "?")
        date = r.get("created_date", "?")[:10] if r.get("created_date") else "?"
        title = r.get("thread_title", "")

        # Show the most informative score
        rerank = r.get("rerank_score")
        rrf = r.get("rrf_score")
        cosine = r.get("score", 0)

        if rerank is not None:
            score_str = f"rerank={rerank:.3f}"
        elif rrf is not None:
            score_str = f"rrf={rrf:.4f}"
        else:
            score_str = f"cos={cosine:.4f}"

        print(f"\n[{i}] {score_str}  |  {tier}  |  r/{sub}  |  {date}")
        print(f"    Thread: {title}")
        print(f"    {'—' * 60}")

        text = r.get("text", "")
        preview = text[:400] + ("…" if len(text) > 400 else "")
        for line in preview.split("\n"):
            print(f"    {line}")

        print("-" * 80)


if __name__ == "__main__":
    main()
