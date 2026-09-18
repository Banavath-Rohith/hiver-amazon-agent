"""Small, inspectable multi-cue keyword intent classifier for AmazonHelp.

Design decisions
----------------
* No external ML libraries required — pure-stdlib implementation so the
  code runs on any Python 3.8+ environment without setup steps.
* Six canonical intents are defined from actual AmazonHelp conversation
  patterns discovered during exploratory analysis.
* Confidence is normalised over competing-keyword overlap so it stays
  in [0, 1] and degrades gracefully when a message activates many buckets.
* Non-English messages are tagged 'other_or_unclear'; the detector uses a
  lightweight Unicode-range heuristic rather than a language library.
"""
from __future__ import annotations

import re
from collections import Counter

# --- Canonical intent taxonomy ------------------------------------------------
INTENTS: dict[str, str] = {
    "delivery_or_order": (
        "order delivery deliver package parcel shipment shipped shipping "
        "delayed delay track tracking arrived dispatch dispatched missing "
        "lost late estimated status where received"
    ),
    "account_or_login": (
        "account login sign password access verify verification "
        "locked lock suspended suspend email username credentials"
    ),
    "payment_or_refund": (
        "refund charge charged payment pay card credit debit money "
        "billing bill invoice overcharged fee return cancel cancelled"
    ),
    "product_or_service": (
        "product item purchase subscribe subscription prime kindle echo "
        "alexa fire tablet device quality broken damaged defective"
    ),
    "technical_issue": (
        "app website site error bug working crash loading glitch button "
        "link issue problem unable open"
    ),
    "other_or_unclear": "",
}

# Characters outside basic ASCII/Latin-Extended-A -> likely non-English
_NON_ASCII_HEAVY = re.compile(r"[\u2e80-\uffef]|[\u00c0-\u024f]")


def tokens(text: str) -> set[str]:
    """Lower-case word tokens from *text*."""
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _is_non_english(text: str) -> bool:
    """Quick heuristic: >10 % of characters are outside ASCII range."""
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / max(len(text), 1)) > 0.10


def classify(message: str) -> dict:
    """Classify *message* and return a detailed result dict.

    Returns
    -------
    dict with keys:
        intent      – one of the six canonical labels
        confidence  – float in [0, 1]
        scores      – raw keyword-overlap counts per intent
        triggered   – list of keywords that fired
    """
    if _is_non_english(message):
        return {
            "intent": "other_or_unclear",
            "confidence": 0.0,
            "scores": {k: 0 for k in INTENTS if k != "other_or_unclear"},
            "triggered": [],
        }

    words = tokens(message)
    # Build per-intent keyword sets (cached once)
    intent_tokens: dict[str, set[str]] = {
        name: tokens(cues) for name, cues in INTENTS.items() if cues
    }

    scores: dict[str, int] = {
        name: len(words & kw_set) for name, kw_set in intent_tokens.items()
    }

    best_intent = max(scores, key=scores.get)
    best_score  = scores[best_intent]

    if best_score == 0:
        return {
            "intent": "other_or_unclear",
            "confidence": 0.0,
            "scores": scores,
            "triggered": [],
        }

    # Normalise: how dominant is the winning bucket vs. all fired keywords?
    all_kw       = set().union(*intent_tokens.values())
    total_fired  = len(words & all_kw)
    confidence   = round(best_score / max(total_fired, 1), 3)

    # Collect the actual keywords that matched
    triggered = sorted(words & intent_tokens[best_intent])

    return {
        "intent": best_intent,
        "confidence": confidence,
        "scores": scores,
        "triggered": triggered,
    }
