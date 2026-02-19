"""Three-tier hierarchical chunker for Reddit threads.

Tier 1 – summary   : thread title + body + metadata  (navigation index)
Tier 2 – chain     : top-level comment + reply tree   (primary retrieval)
Tier 3 – insight   : standalone comment with data     (atomic facts)
"""

import re
from datetime import datetime

from .models import Chunk, Comment, Thread

# Matches financial figures: ₹10K, 15.2%, Rs 500, 12 lakh, CAGR, XIRR …
_NUM_RE = re.compile(
    r"[₹$]\s*[\d,]+|[\d.]+\s*%|[\d.]+\s*(?:lakh|crore|k\b|L\b)",
    re.IGNORECASE,
)

_INSIGHT_TERMS = (
    "better than", "worse than", "i recommend", "my experience",
    "in my opinion", "imo", "compared to", "returns", "cagr",
    "xirr", "sip", "lumpsum", "portfolio", "expense ratio",
    "nav", "aum", "nifty", "sensex",
)


class RedditChunker:
    """Convert Reddit threads into three-tier chunks for embedding."""

    def __init__(
        self,
        min_score: int = 3,
        min_chain_chars: int = 40,
        min_insight_chars: int = 80,
        max_chain_depth: int = 4,
        max_chain_chars: int = 1500,
    ):
        self.min_score = min_score
        self.min_chain_chars = min_chain_chars
        self.min_insight_chars = min_insight_chars
        self.max_chain_depth = max_chain_depth
        self.max_chain_chars = max_chain_chars

    # -- Public --------------------------------------------------------------

    def chunk_threads(self, threads: list[Thread]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for t in threads:
            chunks.extend(self.chunk_thread(t))
        return chunks

    def chunk_thread(self, thread: Thread) -> list[Chunk]:
        chunks: list[Chunk] = []

        summary = self._summary(thread)
        if summary:
            chunks.append(summary)

        chunks.extend(self._chains(thread))
        chunks.extend(self._insights(thread))
        return chunks

    # -- Tier 1: summary ----------------------------------------------------

    def _summary(self, t: Thread) -> Chunk | None:
        date = datetime.fromtimestamp(t.created_utc).strftime("%Y-%m-%d")
        body = t.body[:500] + ("…" if len(t.body) > 500 else "")

        parts = [
            f"Title: {t.title}",
            f"Subreddit: r/{t.subreddit}",
            f"Score: {t.score} | Comments: {t.num_comments} | Date: {date}",
        ]
        if body.strip():
            parts.append(f"\n{body}")

        return Chunk(
            id=f"s_{t.id}",
            text="\n".join(parts),
            tier="summary",
            source_type="reddit",
            metadata=self._thread_meta(t),
        )

    # -- Tier 2: comment chains ----------------------------------------------

    def _chains(self, thread: Thread) -> list[Chunk]:
        chunks: list[Chunk] = []

        for comment in thread.comments:
            if comment.score < self.min_score:
                continue

            flat = comment.flatten(max_depth=self.max_chain_depth)
            lines: list[str] = []
            total = 0

            for c in flat:
                indent = "  " * min(c.depth, 3)
                arrow = "→ " if c.depth > 0 else ""
                line = f"{indent}{arrow}[Score:{c.score}] {c.body}"

                if total + len(line) > self.max_chain_chars:
                    break
                lines.append(line)
                total += len(line)

            text = "\n".join(lines)
            if len(text) < self.min_chain_chars:
                continue

            chain_score = sum(c.score for c in flat)

            meta = self._thread_meta(thread)
            meta.update(
                parent_chunk_id=f"s_{thread.id}",
                root_comment_id=comment.id,
                root_score=comment.score,
                chain_score=chain_score,
                chain_length=len(flat),
                author_count=len({c.author for c in flat}),
            )

            chunks.append(Chunk(
                id=f"c_{thread.id}_{comment.id}",
                text=text,
                tier="chain",
                source_type="reddit",
                metadata=meta,
            ))

        return chunks

    # -- Tier 3: individual insights -----------------------------------------

    def _insights(self, thread: Thread) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen: set[str] = set()

        for top in thread.comments:
            for c in top.flatten():
                if c.id in seen or c.score < self.min_score:
                    continue
                seen.add(c.id)

                if len(c.body) < self.min_insight_chars:
                    continue
                if not self._qualifies_as_insight(c.body):
                    continue

                meta = self._thread_meta(thread)
                meta.update(
                    parent_chunk_id=f"s_{thread.id}",
                    comment_score=c.score,
                    author=c.author,
                    has_numbers=bool(_NUM_RE.search(c.body)),
                )

                chunks.append(Chunk(
                    id=f"i_{thread.id}_{c.id}",
                    text=c.body,
                    tier="insight",
                    source_type="reddit",
                    metadata=meta,
                ))

        return chunks

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _thread_meta(t: Thread) -> dict:
        return {
            "subreddit": t.subreddit,
            "thread_id": t.id,
            "thread_title": t.title,
            "thread_url": t.url,
            "score": t.score,
            "num_comments": t.num_comments,
            "created_date": datetime.fromtimestamp(t.created_utc).isoformat(),
        }

    @staticmethod
    def _qualifies_as_insight(text: str) -> bool:
        if _NUM_RE.search(text):
            return True
        if len(text.split()) > 50:
            return True
        low = text.lower()
        return any(t in low for t in _INSIGHT_TERMS)
