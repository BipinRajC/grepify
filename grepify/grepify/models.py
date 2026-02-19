"""Core data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Comment:
    id: str
    body: str
    score: int
    author: str
    created_utc: float
    parent_id: str | None = None
    depth: int = 0
    replies: list[Comment] = field(default_factory=list)

    @property
    def created_date(self) -> datetime:
        return datetime.fromtimestamp(self.created_utc)

    def flatten(self, max_depth: int = 99) -> list[Comment]:
        """Flatten this comment tree depth-first."""
        result = [self]
        if self.depth < max_depth:
            for r in self.replies:
                result.extend(r.flatten(max_depth))
        return result


@dataclass
class Thread:
    id: str
    subreddit: str
    title: str
    body: str
    score: int
    author: str
    created_utc: float
    url: str
    num_comments: int
    comments: list[Comment] = field(default_factory=list)

    @property
    def created_date(self) -> datetime:
        return datetime.fromtimestamp(self.created_utc)


@dataclass
class Chunk:
    id: str
    text: str
    tier: str            # "summary" | "chain" | "insight"
    source_type: str     # "reddit" | "whatsapp"
    metadata: dict = field(default_factory=dict)
