"""Retrieval pipeline: BM25 + dense search → RRF fusion → MMR → reranking.

This is the core intelligence layer. It takes a raw query and returns
the best, most diverse, most relevant chunks from the index.
"""

import math
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .embedder import Embedder
from .store import QdrantStore


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class Retriever:
    """Four-phase retrieval: dense + BM25 → RRF → MMR → rerank."""

    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        collection: str = "grepify_finance",
    ):
        self.store = store
        self.embedder = embedder
        self.collection = collection

        # BM25 corpus — built lazily on first query
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []

    # -- Public API ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 20,
        bm25_k: int = 20,
        mmr_lambda: float = 0.7,
        rerank: bool = True,
    ) -> list[dict]:
        """
        Full retrieval pipeline.

        1. Dense search (Qdrant) → dense_k candidates
        2. BM25 keyword search    → bm25_k candidates
        3. RRF fusion             → merge into one ranked list
        4. MMR diversification    → select top_k diverse results
        5. Cross-encoder rerank   → final ordering (optional)
        """
        query_vec = self.embedder.embed_query(query)

        # Phase 1+2: parallel retrieval
        dense_hits = self.store.search(
            self.collection, query_vec, limit=dense_k
        )
        bm25_hits = self._bm25_search(query, limit=bm25_k)

        # Phase 3: fuse
        fused = reciprocal_rank_fusion(dense_hits, bm25_hits, k=60)

        # Phase 4: diversify
        if len(fused) > top_k:
            fused = self._mmr(query_vec, fused, top_k, lam=mmr_lambda)

        # Phase 5: rerank with cross-encoder (optional)
        if rerank and len(fused) > 1:
            fused = self._rerank(query, fused)

        return fused[:top_k]

    # -- BM25 ----------------------------------------------------------------

    def _ensure_bm25(self) -> None:
        """Build BM25 index from all documents in the collection."""
        if self._bm25 is not None:
            return

        # Pull all docs from qdrant
        all_points = []
        offset = None
        while True:
            result = self.store.client.scroll(
                collection_name=self.collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        self._bm25_docs = [
            {"score": 0.0, **p.payload} for p in all_points
        ]
        corpus = [tokenize(d.get("text", "")) for d in self._bm25_docs]
        self._bm25 = BM25Okapi(corpus)

    def _bm25_search(self, query: str, limit: int = 20) -> list[dict]:
        self._ensure_bm25()
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            zip(scores, self._bm25_docs),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, doc in ranked[:limit]:
            if score <= 0:
                break
            results.append({**doc, "score": float(score)})
        return results

    # -- MMR -----------------------------------------------------------------

    def _mmr(
        self,
        query_vec: np.ndarray,
        candidates: list[dict],
        k: int,
        lam: float = 0.7,
    ) -> list[dict]:
        """Maximal Marginal Relevance — balance relevance and diversity."""
        # Get embeddings for candidates
        texts = [c.get("text", "") for c in candidates]
        doc_vecs = self.embedder.embed_passages(texts, batch_size=32)

        query_vec = query_vec.reshape(1, -1)
        rel_scores = (doc_vecs @ query_vec.T).flatten()

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                relevance = rel_scores[idx]

                # Max similarity to already-selected docs
                if selected:
                    sel_vecs = doc_vecs[selected]
                    sim_to_selected = float((doc_vecs[idx] @ sel_vecs.T).max())
                else:
                    sim_to_selected = 0.0

                mmr_score = lam * relevance - (1 - lam) * sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected]

    # -- Reranking -----------------------------------------------------------

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rerank using cross-encoder (sentence-transformers CrossEncoder)."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return candidates

        # Lazy-load the reranker
        if not hasattr(self, "_cross_encoder"):
            self._cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=self.embedder.model.device.type,
            )

        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self._cross_encoder.predict(pairs)

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


# -- Fusion utils ------------------------------------------------------------

def reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = 60,
    id_key: str = "chunk_id",
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum( 1 / (k + rank_i) ) for each list where the doc appears.
    k=60 is the standard constant from the original paper.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            doc_id = doc.get(id_key, doc.get("text", "")[:50])
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, rrf_score in ranked:
        doc = docs[doc_id]
        doc["rrf_score"] = rrf_score
        results.append(doc)

    return results
