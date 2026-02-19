# Grepify — Progress Report (Session 1)

**Date:** February 20, 2026

## Project Overview

Grepify is a RAG-powered search engine that answers financial questions using authentic Reddit discussions (and eventually WhatsApp channel data from ISFL — Indian School of Financial Literacy).

**USP:** Unfiltered investor sentiment + expert opinion, not SEO-polluted content. Every answer cites actual Reddit threads.

**Niche:** Indian personal finance — mutual funds, stocks, geopolitics affecting markets.

---

## Architecture Decisions Made

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| RAG vs Fine-tuning | RAG | Works with small datasets, sources are traceable, cheaper, easy to update |
| Embedding model | `intfloat/e5-base-v2` (768d) | Trained on Reddit/StackExchange data (CCPairs), naturally aligned with our data |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lightweight, runs on GPU, good precision |
| Vector DB | Qdrant (local Docker) | Free, fast, good API |
| Backend API | Go (Fiber) + Python sidecar (FastAPI) | Go for HTTP concurrency, Python for ML ops |
| Frontend | SvelteKit (later) | Lightweight, SSR for SEO |
| LLM | Agnostic (Groq/Gemini/OpenAI) | Build interface, swap providers via config |
| Chunking | Three-tier hierarchical | Matches Reddit's nested structure |

## Tech Stack

- **Data pipeline (offline):** Python — scraping, chunking, embedding, Qdrant upload
- **ML sidecar (online):** Python FastAPI — query embedding + cross-encoder reranking
- **API server (online):** Go Fiber — handles queries, calls Qdrant + sidecar + LLM
- **Vector store:** Qdrant (Docker, localhost:6333)
- **Hardware:** 16GB RAM, i7-11370H, RTX 3060, Arch Linux

---

## Retrieval Pipeline (Four Phases)

```
Query → Dense search (Qdrant/E5) + BM25 keyword search
      → Reciprocal Rank Fusion (merge both ranked lists)
      → MMR diversification (force results from different threads)
      → Cross-encoder reranking (precision scoring)
      → Top K results
```

## Three-Tier Chunking Strategy

- **Tier 1 (summary):** Thread title + metadata — navigation index
- **Tier 2 (chain):** Top comment + reply tree — primary retrieval unit
- **Tier 3 (insight):** Standalone data-rich comment — atomic facts

---

## What's Been Built (Phase 1 + Phase 2)

### Files Created

```
grepify/
├── .gitignore
├── .env / .env.example
├── pyproject.toml
├── Makefile
├── grepify/
│   ├── __init__.py
│   ├── models.py          # Thread, Comment, Chunk dataclasses
│   ├── scraper.py         # Reddit .json endpoint scraper
│   ├── chunker.py         # Three-tier hierarchical chunker
│   ├── embedder.py        # E5 embedding (GPU, passage:/query: prefixes)
│   ├── store.py           # Qdrant upload + search (query_points API)
│   └── retriever.py       # BM25 + dense → RRF → MMR → reranker
├── scripts/
│   ├── scrape.py          # CLI: scrape subreddit → JSON
│   ├── index.py           # CLI: JSON → chunk → embed → Qdrant
│   └── search.py          # CLI: query → hybrid retrieval → results
└── data/
    └── raw/reddit/        # Scraped JSON files (gitignored)
```

### Phase 1 (Complete): Core Pipeline
- Reddit scraper via `.json` endpoints (no API key needed)
- Three-tier chunker producing summaries, chains, insights
- E5 embedding generation on RTX 3060
- Qdrant local Docker upload + search
- Test: 5 threads from r/IndiaInvestments → 163 chunks → searchable

### Phase 2 (Complete): Retrieval Quality
- BM25 sparse index (keyword search for exact fund names/tickers)
- Reciprocal Rank Fusion merging dense + sparse results
- MMR diversification (forces results from different threads)
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- Search script supports `--mode hybrid/dense/bm25`, `--no-rerank`, `--mmr-lambda`

### Test Results
- BM25 correctly finds "UTI Nifty Index fund" by exact keyword match
- Hybrid mode pulls results from different threads (MMR working)
- Reranker orders results by true relevance (higher rerank score = better)
- Limited by only 5 threads — need more data for meaningful diversity

---

## Known Issues / Notes

- `qdrant-client` v1.17 uses `query_points()` not `search()` — already fixed in store.py
- E5 model shows `embeddings.position_ids | UNEXPECTED` warning — harmless, can be ignored
- Pylance shows yellow import errors — fixed by selecting venv interpreter in VS Code
- With only 5 threads, most results cluster around the "beginner's guide" thread

---

## What's Next

### Immediate: Scrape Real Dataset
```bash
python scripts/scrape.py IndiaInvestments --sort top --time all --limit 50
python scripts/scrape.py IndiaInvestments --sort top --time year --limit 30
python scripts/scrape.py FIREIndia --sort top --time all --limit 30
python scripts/index.py
```
Re-test retrieval quality with more data.

### Phase 3: API Layer
- Python FastAPI sidecar (embed + rerank endpoints)
- Go Fiber API server
- LLM-agnostic client (Groq/Gemini/OpenAI interface)
- Finance-specific prompt templates with source citation

### Phase 4: Frontend
- SvelteKit search interface
- Results display with citations + thread links

### Future: ISFL WhatsApp Data
- Already have 163 ISFL posts in `/home/bipin/reddit-project/ISOFL/ISOFL-data.json`
- Preprocessor exists at `ISOFL/preprocess_whatsapp_final.py`
- Will be added as "expert opinion" layer alongside Reddit crowd sentiment
- Same pipeline: parse → chunk → embed → upload to same Qdrant collection with `source_type: "whatsapp"`

---

## Key Research Reference

`/home/bipin/reddit-project/RAG-concept.md` contains the full research doc covering:
- Recursive Semantic Chunking (RSC)
- Hierarchical indexing parent-child patterns
- Hybrid retrieval (dense + BM25 + RRF)
- MMR for diversity
- Cross-encoder reranking
- Agentic RAG patterns (LangGraph — deferred to later phase)
- Vector DB benchmarks (Qdrant, Milvus, pgvectorscale)
- Embedding model comparisons (MTEB leaderboard)

---

## Commands Reference

```bash
# Activate venv
cd /home/bipin/reddit-project/grepify
source .venv/bin/activate.fish

# Start Qdrant
docker start qdrant  # or docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Scrape
python scripts/scrape.py IndiaInvestments --sort top --time all --limit 50

# Build index
python scripts/index.py

# Search
python scripts/search.py "best mutual fund for SIP"
python scripts/search.py "UTI Nifty Index fund" --mode bm25
python scripts/search.py "credit card fraud" --no-rerank
python scripts/search.py "investing guide" --mmr-lambda 0.3

# Git (from parent dir)
cd /home/bipin/reddit-project
git add grepify/
git commit -m "message"
git push
```
