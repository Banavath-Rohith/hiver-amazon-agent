"""Unified AI customer-support agent for AmazonHelp.

Pipeline
--------
Customer message
  ↓ intent_classifier.classify()     – intent + confidence + triggered keywords
  ↓ retrieval.retrieve()             – top-3 similar historical cases (Jaccard)
  ↓ reply_generator.draft_reply()   – evidence-grounded verbatim reply
  ↓ escalation.decide()             – AUTO / ESCALATE + stated reason
  ↓ structured JSON output

The agent is intentionally stateless: every call to ``run()`` is fully
reproducible given the same message and case database.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as `python src/agent.py` from the project root
sys.path.insert(0, str(Path(__file__).parent))

from intent_classifier import classify
from retrieval import retrieve
from reply_generator import draft_reply
from escalation import decide


def run(message: str) -> dict:
    """Process a single customer message end-to-end.

    Parameters
    ----------
    message:
        Raw customer tweet text.

    Returns
    -------
    dict with schema:
        intent      – classified intent label
        confidence  – float [0, 1]
        triggered   – keywords that fired the intent
        action      – "AUTO" or "ESCALATE"
        reason      – explanation for the action decision
        reply       – drafted reply text
        evidence    – list of up-to-3 retrieved historical cases
    """
    classification = classify(message)
    # Pass message through for escalation keyword checks
    classification["message"] = message

    evidence = retrieve(message)
    decision = decide(classification, evidence)
    reply    = draft_reply(message, evidence)

    return {
        "intent":     classification["intent"],
        "confidence": classification["confidence"],
        "triggered":  classification.get("triggered", []),
        "action":     decision["action"],
        "reason":     decision["reason"],
        "reply":      reply,
        "evidence":   evidence,
    }


if __name__ == "__main__":
    # Quick smoke-test: pass a message on the command line or use a default.
    msg = " ".join(sys.argv[1:]) or (
        "@AmazonHelp My order hasn't arrived after 10 days, "
        "tracking shows it's been stuck for a week!"
    )
    result = run(msg)
    print(json.dumps(result, indent=2, ensure_ascii=False))
