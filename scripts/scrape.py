#!/usr/bin/env python3
"""Scrape a subreddit and save raw JSON.

Usage:
    python scripts/scrape.py IndiaInvestments --limit 30
    python scripts/scrape.py FIREIndia --sort hot --limit 20
"""

import argparse
from pathlib import Path

from grepify.scraper import RedditScraper


def main():
    ap = argparse.ArgumentParser(description="Scrape a subreddit")
    ap.add_argument("subreddit", help="Subreddit name (without r/)")
    ap.add_argument("--sort", default="top", choices=["top", "hot", "new"])
    ap.add_argument("--time", default="all",
                    choices=["hour", "day", "week", "month", "year", "all"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--no-comments", action="store_true",
                    help="Skip fetching comment trees (faster)")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    scraper = RedditScraper()
    threads = scraper.scrape_subreddit(
        args.subreddit,
        sort=args.sort,
        time_filter=args.time,
        limit=args.limit,
        fetch_comments=not args.no_comments,
    )

    out = args.output or Path(
        f"data/raw/reddit/{args.subreddit}_{args.sort}_{args.time}.json"
    )
    scraper.save(threads, out)


if __name__ == "__main__":
    main()
