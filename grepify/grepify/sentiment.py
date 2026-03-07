"""VADER-based sentiment scoring for Reddit text chunks.

Returns compound scores in [-1.0, +1.0]:
  -1.0 = extremely negative
   0.0 = neutral
  +1.0 = extremely positive
"""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_text(text: str) -> float:
    """Return the VADER compound score for a single text."""
    return _get_analyzer().polarity_scores(text)["compound"]


def score_texts(texts: list[str]) -> list[float]:
    """Return VADER compound scores for a batch of texts."""
    analyzer = _get_analyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]
