"""FastAPI ML sidecar — scrape, chunk, embed, search, sentiment, cluster, upload.

The Go Fiber server calls this sidecar for all ML + Python operations.
Models loaded once at startup; endpoints are stateless after that.

Run with:
    uvicorn api.sidecar:app --host 0.0.0.0 --port 8001 --reload
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COLLECTION   = os.getenv("COLLECTION_NAME",   "grepify_finance")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL",   "intfloat/e5-base-v2")
DEVICE       = os.getenv("EMBEDDING_DEVICE",  "cuda")
QDRANT_URL   = os.getenv("QDRANT_URL",        "http://localhost:6333")
QDRANT_KEY   = os.getenv("QDRANT_API_KEY",    None)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
class _State:
    embedder   = None
    store      = None
    retriever  = None
    scraper    = None
    chunker    = None


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and connections at startup."""
    from grepify.embedder  import Embedder
    from grepify.store     import QdrantStore
    from grepify.retriever import Retriever
    from grepify.scraper   import RedditScraper
    from grepify.chunker   import RedditChunker

    print(f"Loading embedder ({EMBED_MODEL} on {DEVICE})…", flush=True)
    state.embedder  = Embedder(model_name=EMBED_MODEL, device=DEVICE)

    print("Connecting to Qdrant…", flush=True)
    state.store     = QdrantStore(url=QDRANT_URL, api_key=QDRANT_KEY)

    print("Initialising retriever…", flush=True)
    state.retriever = Retriever(
        store=state.store,
        embedder=state.embedder,
        collection=COLLECTION,
    )

    # Warm up BM25 index
    print("Warming up BM25 index…", flush=True)
    state.retriever._ensure_bm25()
    print(f"BM25 corpus: {len(state.retriever._bm25_docs)} docs", flush=True)

    # Warm up cross-encoder
    print("Warming up cross-encoder…", flush=True)
    from sentence_transformers import CrossEncoder
    state.retriever._cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=state.embedder.model.device.type,
    )

    # Scraper + Chunker
    state.scraper = RedditScraper(delay=1.0)
    state.chunker = RedditChunker()

    print("Sidecar ready.", flush=True)
    yield


app = FastAPI(title="Grepify ML Sidecar", version="0.2.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# ---- Pydantic models ----
# ---------------------------------------------------------------------------

# --- Scrape ---

class ScrapeRequest(BaseModel):
    subreddit:   str
    sort:        str = "top"
    time_filter: str = "all"
    limit:       int = Field(50, ge=1, le=500)


class ScrapeComment(BaseModel):
    id:         str
    body:       str
    score:      int
    author:     str
    created_utc: float
    parent_id:  str | None = None
    depth:      int = 0
    replies:    list["ScrapeComment"] = []


class ScrapeThread(BaseModel):
    id:           str
    subreddit:    str
    title:        str
    body:         str
    score:        int
    author:       str
    url:          str
    created_utc:  float
    num_comments: int
    comments:     list[ScrapeComment] = []


class ScrapeResponse(BaseModel):
    threads:    list[ScrapeThread]
    elapsed_ms: float


# --- Chunk ---

class ChunkRequest(BaseModel):
    threads: list[ScrapeThread]


class ChunkResult(BaseModel):
    chunk_id:     str
    text:         str
    tier:         str
    thread_id:    str
    thread_title: str
    subreddit:    str
    url:          str
    score:        int
    created_date: str


class ChunkResponse(BaseModel):
    chunks:     list[ChunkResult]
    elapsed_ms: float


# --- Embed ---

class EmbedRequest(BaseModel):
    texts: list[str]
    kind:  Literal["query", "passage"] = "query"


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    elapsed_ms: float


# --- Search ---

class SearchRequest(BaseModel):
    query:      str
    collection: str = ""
    top_k:      int   = Field(5,   ge=1, le=100)
    mode:       Literal["hybrid", "dense", "bm25"] = "hybrid"
    rerank:     bool  = True
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)
    dense_k:    int   = Field(20,  ge=1, le=200)
    bm25_k:     int   = Field(20,  ge=1, le=200)
    min_score:  int   = 0  # karma filter


class SearchResult(BaseModel):
    text:          str
    tier:          str
    thread_id:     str
    thread_title:  str
    subreddit:     str
    url:           str
    score:         float
    rerank_score:  float | None = None


class SearchResponse(BaseModel):
    query:      str
    results:    list[SearchResult]
    elapsed_ms: float


# --- Rerank ---

class RerankRequest(BaseModel):
    query:    str
    passages: list[str]


class RerankResponse(BaseModel):
    scores:     list[float]
    elapsed_ms: float


# --- Sentiment ---

class SentimentRequest(BaseModel):
    texts: list[str]


class SentimentResponse(BaseModel):
    scores:     list[float]
    elapsed_ms: float


# --- Cluster ---

class ClusterRequest(BaseModel):
    texts: list[str]


class ClusterGroup(BaseModel):
    id:    int
    texts: list[str]
    size:  int


class ClusterResponse(BaseModel):
    clusters:   list[ClusterGroup]
    elapsed_ms: float


# --- Upload (batch upsert to Qdrant) ---

class UploadPoint(BaseModel):
    id:      str
    vector:  list[float]
    payload: dict


class UploadRequest(BaseModel):
    collection: str
    points:     list[UploadPoint]


class UploadResponse(BaseModel):
    uploaded:   int
    elapsed_ms: float


# --- Ensure Collection ---

class EnsureCollectionReq(BaseModel):
    name: str
    dim:  int = 768


class EnsureCollectionResp(BaseModel):
    name:    str
    created: bool


# ---------------------------------------------------------------------------
# ---- Endpoints ----
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    bm25_size = len(state.retriever._bm25_docs) if state.retriever else 0
    return {
        "status": "ok",
        "collection": COLLECTION,
        "bm25_docs": bm25_size,
        "device": DEVICE,
    }


# ---- Scrape ----

@app.post("/scrape", response_model=ScrapeResponse)
def scrape(req: ScrapeRequest):
    if state.scraper is None:
        raise HTTPException(503, "Scraper not ready")

    t0 = time.perf_counter()
    threads = state.scraper.scrape_subreddit(
        subreddit=req.subreddit,
        sort=req.sort,
        time_filter=req.time_filter,
        limit=req.limit,
    )

    def _comment_to_dict(c) -> dict:
        return {
            "id": c.id, "body": c.body, "score": c.score,
            "author": c.author, "created_utc": c.created_utc,
            "parent_id": c.parent_id, "depth": c.depth,
            "replies": [_comment_to_dict(r) for r in c.replies],
        }

    out_threads = []
    for t in threads:
        out_threads.append(ScrapeThread(
            id=t.id, subreddit=t.subreddit, title=t.title, body=t.body,
            score=t.score, author=t.author, url=t.url,
            created_utc=t.created_utc, num_comments=t.num_comments,
            comments=[ScrapeComment(**_comment_to_dict(c)) for c in t.comments],
        ))

    elapsed = (time.perf_counter() - t0) * 1000
    return ScrapeResponse(threads=out_threads, elapsed_ms=round(elapsed, 1))


# ---- Chunk ----

def _thread_from_scrape(st: ScrapeThread):
    """Convert sidecar ScrapeThread to grepify.models.Thread."""
    from grepify.models import Thread, Comment as ModelComment

    def _to_model_comment(sc: ScrapeComment) -> ModelComment:
        return ModelComment(
            id=sc.id, body=sc.body, score=sc.score, author=sc.author,
            created_utc=sc.created_utc, parent_id=sc.parent_id, depth=sc.depth,
            replies=[_to_model_comment(r) for r in sc.replies],
        )

    return Thread(
        id=st.id, subreddit=st.subreddit, title=st.title, body=st.body,
        score=st.score, author=st.author, url=st.url,
        created_utc=st.created_utc, num_comments=st.num_comments,
        comments=[_to_model_comment(c) for c in st.comments],
    )


@app.post("/chunk", response_model=ChunkResponse)
def chunk(req: ChunkRequest):
    if state.chunker is None:
        raise HTTPException(503, "Chunker not ready")

    t0 = time.perf_counter()
    model_threads = [_thread_from_scrape(t) for t in req.threads]
    chunks = state.chunker.chunk_threads(model_threads)

    results = []
    for ch in chunks:
        results.append(ChunkResult(
            chunk_id=ch.id,
            text=ch.text,
            tier=ch.tier,
            thread_id=ch.metadata.get("thread_id", ""),
            thread_title=ch.metadata.get("thread_title", ""),
            subreddit=ch.metadata.get("subreddit", ""),
            url=ch.metadata.get("url", ""),
            score=ch.metadata.get("score", 0),
            created_date=ch.metadata.get("created_date", ""),
        ))

    elapsed = (time.perf_counter() - t0) * 1000
    return ChunkResponse(chunks=results, elapsed_ms=round(elapsed, 1))


# ---- Search ----

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if state.retriever is None:
        raise HTTPException(503, "Retriever not ready")

    t0 = time.perf_counter()
    retriever = state.retriever
    collection = req.collection or COLLECTION

    if req.mode == "dense":
        query_vec  = retriever.embedder.embed_query(req.query)
        dense_hits = retriever.store.search(collection, query_vec, limit=req.dense_k)
        from grepify.retriever import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(dense_hits, [], k=60)
    elif req.mode == "bm25":
        fused = retriever._bm25_search(req.query, limit=req.bm25_k)
    else:  # hybrid
        query_vec  = retriever.embedder.embed_query(req.query)
        dense_hits = retriever.store.search(collection, query_vec, limit=req.dense_k)
        bm25_hits  = retriever._bm25_search(req.query, limit=req.bm25_k)
        from grepify.retriever import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(dense_hits, bm25_hits, k=60)

    # Karma filter
    if req.min_score > 0:
        fused = [d for d in fused if d.get("score", 0) >= req.min_score]

    # MMR
    if len(fused) > req.top_k and req.mode != "bm25":
        query_vec = retriever.embedder.embed_query(req.query)
        fused = retriever._mmr(query_vec, fused, req.top_k, lam=req.mmr_lambda)

    # Rerank
    if req.rerank and len(fused) > 1:
        fused = retriever._rerank(req.query, fused)

    results = []
    for doc in fused[:req.top_k]:
        results.append(SearchResult(
            text         = doc.get("text", ""),
            tier         = doc.get("tier", ""),
            thread_id    = doc.get("thread_id", ""),
            thread_title = doc.get("thread_title", ""),
            subreddit    = doc.get("subreddit", ""),
            url          = doc.get("url", ""),
            score        = float(doc.get("rrf_score", doc.get("score", 0.0))),
            rerank_score = doc.get("rerank_score"),
        ))

    elapsed = (time.perf_counter() - t0) * 1000
    return SearchResponse(query=req.query, results=results, elapsed_ms=round(elapsed, 1))


# ---- Embed ----

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if state.embedder is None:
        raise HTTPException(503, "Embedder not ready")

    t0 = time.perf_counter()
    if req.kind == "query":
        vecs = np.array([state.embedder.embed_query(t) for t in req.texts])
    else:
        vecs = state.embedder.embed_passages(req.texts)

    elapsed = (time.perf_counter() - t0) * 1000
    return EmbedResponse(
        embeddings=vecs.tolist(),
        elapsed_ms=round(elapsed, 1),
    )


# ---- Rerank ----

@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if state.retriever is None:
        raise HTTPException(503, "Retriever not ready")

    t0 = time.perf_counter()
    ce = state.retriever._cross_encoder
    pairs  = [(req.query, p) for p in req.passages]
    scores = ce.predict(pairs).tolist()

    elapsed = (time.perf_counter() - t0) * 1000
    return RerankResponse(scores=scores, elapsed_ms=round(elapsed, 1))


# ---- Sentiment ----

@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest):
    t0 = time.perf_counter()
    from grepify.sentiment import score_texts
    scores = score_texts(req.texts)
    elapsed = (time.perf_counter() - t0) * 1000
    return SentimentResponse(scores=scores, elapsed_ms=round(elapsed, 1))


# ---- Cluster ----

@app.post("/cluster", response_model=ClusterResponse)
def cluster(req: ClusterRequest):
    if len(req.texts) < 5:
        return ClusterResponse(
            clusters=[ClusterGroup(id=0, texts=req.texts, size=len(req.texts))],
            elapsed_ms=0.0,
        )

    t0 = time.perf_counter()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        import math

        n_clusters = min(max(2, int(math.sqrt(len(req.texts)))), 8)
        tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
        X = tfidf.fit_transform(req.texts)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        groups: dict[int, list[str]] = {}
        for label, text in zip(labels, req.texts):
            groups.setdefault(int(label), []).append(text)

        clusters = [
            ClusterGroup(id=k, texts=v, size=len(v))
            for k, v in sorted(groups.items())
        ]
    except Exception as e:
        print(f"Clustering failed: {e}", flush=True)
        clusters = [ClusterGroup(id=0, texts=req.texts, size=len(req.texts))]

    elapsed = (time.perf_counter() - t0) * 1000
    return ClusterResponse(clusters=clusters, elapsed_ms=round(elapsed, 1))


# ---- Upload ----

@app.post("/upload", response_model=UploadResponse)
def upload(req: UploadRequest):
    if state.store is None:
        raise HTTPException(503, "Store not ready")

    t0 = time.perf_counter()
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(id=p.id, vector=p.vector, payload=p.payload)
        for p in req.points
    ]

    BATCH = 100
    for i in range(0, len(points), BATCH):
        batch = points[i:i + BATCH]
        state.store.client.upsert(collection_name=req.collection, points=batch)

    elapsed = (time.perf_counter() - t0) * 1000
    return UploadResponse(uploaded=len(points), elapsed_ms=round(elapsed, 1))


# ---- Ensure Collection ----

@app.post("/collections/ensure", response_model=EnsureCollectionResp)
def ensure_collection(req: EnsureCollectionReq):
    if state.store is None:
        raise HTTPException(503, "Store not ready")

    from qdrant_client.models import VectorParams, Distance

    try:
        info = state.store.client.get_collection(req.name)
        return EnsureCollectionResp(name=req.name, created=False)
    except Exception:
        state.store.client.create_collection(
            collection_name=req.name,
            vectors_config=VectorParams(size=req.dim, distance=Distance.COSINE),
        )
        return EnsureCollectionResp(name=req.name, created=True)
