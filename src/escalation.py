"""Conservative escalation rules for the AmazonHelp AI agent.

IMPORTANT – these are SYSTEM-DESIGN decisions, not Amazon policy.
-----------------------------------------------------------------
The rules below are chosen to minimise harm from an automated agent
acting on sensitive customer data.  They are documented in decision_log.md
(items 7–9) and should be adjusted by a domain expert before production use.

Rule priority (evaluated in order):
    1. Sensitive keywords -> ESCALATE  (security / privacy / financial)
    2. Non-English / unclear intent   -> ESCALATE  (language barrier risk)
    3. Technical issues               -> ESCALATE  (requires system access)
    4. Low classifier confidence      -> ESCALATE  (ambiguous query)
    5. No historical evidence         -> ESCALATE  (no grounding available)
    6. Otherwise                      -> AUTO
"""
from __future__ import annotations

import re

# Keywords that signal a sensitive interaction requiring human review.
SENSITIVE_KEYWORDS: frozenset[str] = frozenset({
    "card", "charge", "charged", "payment", "refund",
    "account", "login", "password", "security", "billing",
    "overcharged", "credentials", "locked", "suspended", "unauthorized",
})

LOW_CONFIDENCE_THRESHOLD = 0.15


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def decide(classification: dict, evidence: list[dict]) -> dict:
    """Return an escalation decision dict with keys ``action`` and ``reason``.

    Parameters
    ----------
    classification:
        Output of ``intent_classifier.classify()``.
        Must include keys: ``intent``, ``confidence``, ``message`` (optional).
    evidence:
        Output of ``retrieval.retrieve()``.

    Returns
    -------
    dict with:
        action – "AUTO" or "ESCALATE"
        reason – human-readable explanation
    """
    message = classification.get("message", "")
    intent  = classification.get("intent", "other_or_unclear")
    conf    = classification.get("confidence", 0.0)

    # Rule 1: Sensitive keywords
    if _tokens(message) & SENSITIVE_KEYWORDS:
        return {
            "action": "ESCALATE",
            "reason": (
                "Message contains account, payment, or security-sensitive "
                "keywords that require human verification before action."
            ),
        }

    # Rule 2: Non-English / unclear intent
    if intent == "other_or_unclear":
        return {
            "action": "ESCALATE",
            "reason": (
                "Intent is unclear or the message is in a non-English language; "
                "a human agent should handle this to avoid misunderstanding."
            ),
        }

    # Rule 3: Technical issues require system access
    if intent == "technical_issue":
        return {
            "action": "ESCALATE",
            "reason": (
                "Technical issues typically require access to account or system "
                "context that the automated agent cannot safely retrieve."
            ),
        }

    # Rule 4: Low classifier confidence
    if conf < LOW_CONFIDENCE_THRESHOLD:
        return {
            "action": "ESCALATE",
            "reason": (
                f"Low intent confidence ({conf:.0%}); the agent is not "
                "sufficiently certain about the customer's request."
            ),
        }

    # Rule 5: No historical evidence to ground a reply
    if not evidence:
        return {
            "action": "ESCALATE",
            "reason": (
                "No similar historical AmazonHelp case was found; "
                "a human agent should respond to avoid an unsupported reply."
            ),
        }

    # Rule 6: All checks passed -> auto-handle
    return {
        "action": "AUTO",
        "reason": (
            f"Clear '{intent}' intent (confidence {conf:.0%}), "
            "sensitive-keyword check passed, and historical evidence is available."
        ),
    }
