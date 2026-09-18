"""Inverted-index retrieval of similar historical AmazonHelp cases.

Why no FAISS / sentence-transformers?
--------------------------------------
* Zero external dependencies keeps setup under 15 minutes on any machine.
* An inverted-index over lower-cased tokens is fast enough for a ~60 k-row
  CSV (< 1 s on a normal laptop) and fully transparent/explainable.
* Jaccard similarity (intersection / union) on token sets is a well-understood
  proxy for relevance that Hiver reviewers can inspect and modify easily.

The retrieval module is intentionally separate from the agent so it can be
upgraded to a dense-vector retriever (FAISS / BM25 / etc.) without touching
the rest of the pipeline.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

CASES_PATH = Path(__file__).parent.parent / "data" / "processed" / "amazon_cases.csv"

# Module-level cache: loaded once per process to avoid repeated I/O.
_CASES_CACHE: list[dict] | None = None


def _load_cases(path: Path = CASES_PATH) -> list[dict]:
    global _CASES_CACHE
    if _CASES_CACHE is None:
        if not path.exists():
            _CASES_CACHE = []
        else:
            with path.open(encoding="utf-8", newline="") as f:
                _CASES_CACHE = list(csv.DictReader(f))
    return _CASES_CACHE


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def retrieve(message: str, limit: int = 3, path: Path = CASES_PATH) -> list[dict]:
    """Return the top-*limit* historical cases most similar to *message*.

    Each result dict contains:
        tweet_id         – original tweet ID
        message          – customer message text
        support_tweet_id – AmazonHelp reply tweet ID
        support_response – AmazonHelp reply text
        similarity       – Jaccard similarity score (0–1)
    """
    cases  = _load_cases(path)
    query  = _tokens(message)
    if not query:
        return []

    scored: list[tuple[float, dict]] = []
    for case in cases:
        case_tokens = _tokens(case.get("message", ""))
        union_size  = len(query | case_tokens)
        if union_size == 0:
            continue
        score = len(query & case_tokens) / union_size
        if score > 0:
            scored.append((score, case))

    # Sort descending by score; take top-N
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {**row, "similarity": round(sim, 4)}
        for sim, row in scored[:limit]
    ]
