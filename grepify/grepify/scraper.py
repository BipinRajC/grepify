"""Reddit scraper using public .json endpoints. No API keys needed."""

import json
import time
from pathlib import Path

import requests

from .models import Comment, Thread


class RedditScraper:
    BASE = "https://www.reddit.com"

    def __init__(self, delay: float = 2.0):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Grepify/1.0"
        self.delay = delay
        self._last_req = 0.0

    # -- HTTP ----------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> dict:
        wait = self.delay - (time.time() - self._last_req)
        if wait > 0:
            time.sleep(wait)

        if not url.endswith(".json"):
            url += ".json"

        resp = self.session.get(url, params=params)
        self._last_req = time.time()
        resp.raise_for_status()
        return resp.json()

    # -- Parsing -------------------------------------------------------------

    def _parse_comment(self, raw: dict, depth: int = 0) -> Comment | None:
        if raw.get("kind") != "t1":
            return None
        d = raw["data"]
        body = d.get("body", "")
        if body in ("[deleted]", "[removed]", ""):
            return None

        comment = Comment(
            id=d.get("id", ""),
            body=body,
            score=d.get("score", 0),
            author=d.get("author", "[deleted]"),
            created_utc=d.get("created_utc", 0),
            parent_id=d.get("parent_id"),
            depth=depth,
        )

        replies_data = d.get("replies")
        if isinstance(replies_data, dict):
            for child in replies_data.get("data", {}).get("children", []):
                reply = self._parse_comment(child, depth + 1)
                if reply:
                    comment.replies.append(reply)

        return comment

    def _parse_thread(self, raw: dict, subreddit: str) -> Thread | None:
        if raw.get("kind") != "t3":
            return None
        d = raw["data"]
        return Thread(
            id=d.get("id", ""),
            subreddit=subreddit,
            title=d.get("title", ""),
            body=d.get("selftext", ""),
            score=d.get("score", 0),
            author=d.get("author", "[deleted]"),
            created_utc=d.get("created_utc", 0),
            url=f"https://reddit.com{d.get('permalink', '')}",
            num_comments=d.get("num_comments", 0),
        )

    # -- Public API ----------------------------------------------------------

    def get_listing(
        self,
        subreddit: str,
        sort: str = "top",
        time_filter: str = "all",
        limit: int = 100,
    ) -> list[Thread]:
        """Fetch thread stubs from a subreddit listing page."""
        threads: list[Thread] = []
        after = None

        while len(threads) < limit:
            params = {
                "limit": min(100, limit - len(threads)),
                "t": time_filter,
                "raw_json": 1,
            }
            if after:
                params["after"] = after

            data = self._get(f"{self.BASE}/r/{subreddit}/{sort}", params)
            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                t = self._parse_thread(child, subreddit)
                if t:
                    threads.append(t)

            after = data.get("data", {}).get("after")
            if not after:
                break

            print(f"  listing: {len(threads)}/{limit}")

        return threads

    def get_thread_comments(self, url: str) -> Thread | None:
        """Fetch a single thread with its full comment tree."""
        try:
            data = self._get(url)
        except requests.RequestException as e:
            print(f"  error: {e}")
            return None

        if not data or len(data) < 2:
            return None

        listing = data[0].get("data", {}).get("children", [])
        if not listing:
            return None

        raw = listing[0]
        sub = raw.get("data", {}).get("subreddit", "")
        thread = self._parse_thread(raw, sub)
        if not thread:
            return None

        for child in data[1].get("data", {}).get("children", []):
            c = self._parse_comment(child)
            if c:
                thread.comments.append(c)

        return thread

    def scrape_subreddit(
        self,
        subreddit: str,
        sort: str = "top",
        time_filter: str = "all",
        limit: int = 50,
        fetch_comments: bool = True,
    ) -> list[Thread]:
        """Scrape a subreddit: listing + optionally full comments per thread."""
        print(f"scraping r/{subreddit} ({sort}/{time_filter}, limit={limit})")
        threads = self.get_listing(subreddit, sort, time_filter, limit)
        print(f"  got {len(threads)} threads")

        if fetch_comments:
            for i, thread in enumerate(threads):
                full = self.get_thread_comments(thread.url)
                if full:
                    threads[i] = full
                n = len(threads[i].comments)
                print(f"  comments: {i + 1}/{len(threads)} ({n} top-level)")

        return threads

    # -- Persistence ---------------------------------------------------------

    @staticmethod
    def save(threads: list[Thread], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        def _ser_comment(c: Comment) -> dict:
            return {
                "id": c.id,
                "body": c.body,
                "score": c.score,
                "author": c.author,
                "created_utc": c.created_utc,
                "parent_id": c.parent_id,
                "depth": c.depth,
                "replies": [_ser_comment(r) for r in c.replies],
            }

        out = [
            {
                "id": t.id,
                "subreddit": t.subreddit,
                "title": t.title,
                "body": t.body,
                "score": t.score,
                "author": t.author,
                "created_utc": t.created_utc,
                "url": t.url,
                "num_comments": t.num_comments,
                "comments": [_ser_comment(c) for c in t.comments],
            }
            for t in threads
        ]

        path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"saved {len(out)} threads → {path}")

    @staticmethod
    def load(path: Path) -> list[Thread]:
        data = json.loads(path.read_text())

        def _deser_comment(d: dict) -> Comment:
            return Comment(
                id=d["id"],
                body=d["body"],
                score=d["score"],
                author=d["author"],
                created_utc=d["created_utc"],
                parent_id=d.get("parent_id"),
                depth=d.get("depth", 0),
                replies=[_deser_comment(r) for r in d.get("replies", [])],
            )

        return [
            Thread(
                id=d["id"],
                subreddit=d["subreddit"],
                title=d["title"],
                body=d["body"],
                score=d["score"],
                author=d["author"],
                created_utc=d["created_utc"],
                url=d["url"],
                num_comments=d["num_comments"],
                comments=[_deser_comment(c) for c in d.get("comments", [])],
            )
            for d in data
        ]
