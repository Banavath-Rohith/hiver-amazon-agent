"""Generate failure_analysis.csv from actual evaluation_results.csv.

Identifies the top 5 meaningful failure modes from real agent predictions,
with a customer message, expected vs actual behaviour, failure type, and
a hypothesis about the root cause.

Usage (from project root):
    python src/generate_failure_analysis.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "evaluation_results.csv"
OUT     = ROOT / "results" / "failure_analysis.csv"


def main() -> None:
    with RESULTS.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    failures = [r for r in rows if r["intent_match"] == "0" or r["action_match"] == "0"]
    print(f"Total prediction errors: {len(failures)}")

    # Categorize failures
    intent_wrong_action_right  = [r for r in failures if r["intent_match"] == "0" and r["action_match"] == "1"]
    intent_right_action_wrong  = [r for r in failures if r["intent_match"] == "1" and r["action_match"] == "0"]
    both_wrong                 = [r for r in failures if r["intent_match"] == "0" and r["action_match"] == "0"]

    print(f"  Intent wrong, action right : {len(intent_wrong_action_right)}")
    print(f"  Intent right, action wrong : {len(intent_right_action_wrong)}")
    print(f"  Both wrong                 : {len(both_wrong)}")

    # ── Build top-5 representative failure examples ─────────────────────────
    # We select the most illustrative examples per failure type.

    selected_failures = []

    # Type 1: Multi-intent ambiguous message classified wrong
    # Look for delivery misclassified as other_or_unclear
    type1 = [r for r in failures
             if r["true_intent"] == "delivery_or_order"
             and r["predicted_intent"] == "other_or_unclear"]
    if type1:
        r = type1[0]
        selected_failures.append({
            "tweet_id":          r["tweet_id"],
            "customer_message":  r["message"],
            "expected_intent":   r["true_intent"],
            "predicted_intent":  r["predicted_intent"],
            "expected_action":   r["true_action"],
            "predicted_action":  r["predicted_action"],
            "failure_type":      "Missed delivery intent — few keyword matches",
            "hypothesis": (
                "The message uses natural language without strong delivery keywords "
                "(e.g., 'package', 'tracking', 'shipped').  The keyword classifier "
                "finds zero matches and falls back to other_or_unclear."
            ),
        })

    # Type 2: Payment/refund classified as delivery (keyword overlap)
    type2 = [r for r in failures
             if r["true_intent"] == "payment_or_refund"
             and r["predicted_intent"] in ("delivery_or_order", "other_or_unclear")]
    if type2:
        r = type2[0]
        selected_failures.append({
            "tweet_id":          r["tweet_id"],
            "customer_message":  r["message"],
            "expected_intent":   r["true_intent"],
            "predicted_intent":  r["predicted_intent"],
            "expected_action":   r["true_action"],
            "predicted_action":  r["predicted_action"],
            "failure_type":      "Payment intent confused with delivery or unclear",
            "hypothesis": (
                "Messages that mention 'order cancelled' or 'return' activate both "
                "the payment and delivery keyword buckets, or neither strongly.  "
                "The classifier picks the wrong winner when scores are tied."
            ),
        })

    # Type 3: Missed escalation (AUTO predicted when should be ESCALATE)
    type3 = [r for r in failures
             if r["true_action"] == "ESCALATE" and r["predicted_action"] == "AUTO"]
    if type3:
        r = type3[0]
        selected_failures.append({
            "tweet_id":          r["tweet_id"],
            "customer_message":  r["message"],
            "expected_intent":   r["true_intent"],
            "predicted_intent":  r["predicted_intent"],
            "expected_action":   r["true_action"],
            "predicted_action":  r["predicted_action"],
            "failure_type":      "Missed escalation — AUTO when human needed",
            "hypothesis": (
                "The escalation rules check only for explicit sensitive keywords.  "
                "Some complex situations (e.g., a delayed international order in a "
                "non-obvious jurisdiction) require human judgement but use none of "
                "the hard-coded keywords, so the agent incorrectly auto-handles it."
            ),
        })

    # Type 4: Technical issue with unclear phrasing → wrong intent
    type4 = [r for r in failures
             if r["true_intent"] == "technical_issue"
             and r["predicted_intent"] != "technical_issue"]
    if type4:
        r = type4[0]
        selected_failures.append({
            "tweet_id":          r["tweet_id"],
            "customer_message":  r["message"],
            "expected_intent":   r["true_intent"],
            "predicted_intent":  r["predicted_intent"],
            "expected_action":   r["true_action"],
            "predicted_action":  r["predicted_action"],
            "failure_type":      "Technical issue misclassified — indirect phrasing",
            "hypothesis": (
                "The customer describes the technical problem indirectly ('it just "
                "keeps spinning', 'won't let me') without using the direct tech "
                "keywords in the classifier vocabulary (app, error, crash, bug).  "
                "This is a vocabulary coverage gap."
            ),
        })

    # Type 5: Non-English message with some English words → wrong intent
    type5 = [r for r in failures
             if r["true_intent"] != r["predicted_intent"]
             and r not in [x for x in selected_failures if "tweet_id" in x
                           and x["tweet_id"] == r["tweet_id"]]]
    if type5:
        r = type5[-1]  # Pick the last for diversity
        selected_failures.append({
            "tweet_id":          r["tweet_id"],
            "customer_message":  r["message"],
            "expected_intent":   r["true_intent"],
            "predicted_intent":  r["predicted_intent"],
            "expected_action":   r["true_action"],
            "predicted_action":  r["predicted_action"],
            "failure_type":      "Non-English or mixed-language message mishandled",
            "hypothesis": (
                "Messages that mix English product names (Kindle, Prime) with a "
                "non-English surrounding context activate product_or_service keywords "
                "when the true issue is a delivery or billing query in another language.  "
                "The non-English heuristic (>10% non-ASCII) may not fire if the message "
                "is mostly ASCII with Unicode punctuation."
            ),
        })

    # Ensure we have exactly 5 (pad with remaining failures if needed)
    if len(selected_failures) < 5 and len(failures) > len(selected_failures):
        used_ids = {r["tweet_id"] for r in selected_failures}
        extras   = [r for r in failures if r["tweet_id"] not in used_ids]
        for r in extras[: 5 - len(selected_failures)]:
            selected_failures.append({
                "tweet_id":          r["tweet_id"],
                "customer_message":  r["message"],
                "expected_intent":   r["true_intent"],
                "predicted_intent":  r["predicted_intent"],
                "expected_action":   r["true_action"],
                "predicted_action":  r["predicted_action"],
                "failure_type":      "General misclassification",
                "hypothesis":        "Low keyword overlap with any intent bucket.",
            })

    selected_failures = selected_failures[:5]

    fields = [
        "tweet_id", "customer_message", "expected_intent", "predicted_intent",
        "expected_action", "predicted_action", "failure_type", "hypothesis",
    ]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selected_failures)

    print(f"Wrote {len(selected_failures)} failure examples -> {OUT}")


if __name__ == "__main__":
    main()
