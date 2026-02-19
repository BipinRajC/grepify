#!/usr/bin/env python3
"""Test retrieval against the Qdrant index.

Usage:
    python scripts/search.py "best mutual fund for SIP"
    python scripts/search.py "defense stocks geopolitics" --tier chain --limit 3
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from grepify.embedder import Embedder
from grepify.store import QdrantStore


def main():
    ap = argparse.ArgumentParser(description="Search the index")
    ap.add_argument("query", help="Search query")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--tier", default=None,
                    choices=["summary", "chain", "insight"])
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

    vec = embedder.embed_query(args.query)
    results = store.search(args.collection, vec, limit=args.limit, tier=args.tier)

    print(f"\nQuery: {args.query}")
    print(f"Results: {len(results)}")
    print("=" * 80)

    for i, r in enumerate(results, 1):
        tier = r.get("tier", "?")
        score = r.get("score", 0)
        sub = r.get("subreddit", "?")
        date = r.get("created_date", "?")[:10] if r.get("created_date") else "?"
        title = r.get("thread_title", "")

        print(f"\n[{i}] {score:.4f}  |  {tier}  |  r/{sub}  |  {date}")
        print(f"    Thread: {title}")
        print(f"    {'—' * 60}")

        text = r.get("text", "")
        preview = text[:400] + ("…" if len(text) > 400 else "")
        for line in preview.split("\n"):
            print(f"    {line}")

        print("-" * 80)


if __name__ == "__main__":
    main()
