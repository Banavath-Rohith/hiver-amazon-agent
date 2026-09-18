"""Evidence-constrained reply generation for AmazonHelp.

Design principle (Decision Log item 6)
----------------------------------------
The generator NEVER invents a resolution.  It picks the best verbatim
historical support response from the retrieved evidence and presents it.
This prevents hallucinated policy claims (e.g., "Your refund will arrive
in 3 days") that could mislead customers or create legal risk.

When no usable evidence is available, the system escalates with a neutral
holding message rather than fabricating an answer.
"""
from __future__ import annotations


ESCALATION_MESSAGE = (
    "Thank you for reaching out to AmazonHelp. "
    "A support specialist will review your case and get back to you shortly."
)


def draft_reply(message: str, evidence: list[dict]) -> str:
    """Draft a grounded reply for *message* using *evidence*.

    Parameters
    ----------
    message:
        The raw customer tweet.
    evidence:
        List of retrieved historical case dicts (each has a
        ``support_response`` key).

    Returns
    -------
    str – the best available verbatim historical response, or the standard
    escalation holding message if no suitable response exists.
    """
    if not evidence:
        return ESCALATION_MESSAGE

    # Prefer the highest-similarity response that is non-empty
    for case in evidence:
        resp = (case.get("support_response") or "").strip()
        if resp:
            return resp

    return ESCALATION_MESSAGE
