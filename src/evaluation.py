"""Full evaluation of the AI agent on the human-labeled golden evaluation set.

Metrics computed
----------------
Intent classification
  - Accuracy, Macro Precision, Macro Recall, Macro F1
  - Per-intent Precision, Recall, F1
  - Confusion matrix

Escalation decision
  - Accuracy, Precision, Recall, F1
  - False Escalation Rate (AUTO predicted as ESCALATE)
  - Missed Escalation Rate (ESCALATE predicted as AUTO)

Results saved to
  results/metrics.json              -- all numeric metrics
  results/evaluation_results.csv    -- row-by-row predictions

Usage (from project root):
    python src/evaluation.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import run as agent_run

GOLDEN  = ROOT / "data" / "golden" / "golden_set.csv"
OUT_CSV = ROOT / "results" / "evaluation_results.csv"
OUT_JSON = ROOT / "results" / "metrics.json"


# ─── Metric utilities ─────────────────────────────────────────────────────────

def compute_clf_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict:
    n = len(y_true)
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / max(n, 1)
    per_class: dict[str, dict] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[label] = {
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "support":   sum(t == label for t in y_true),
        }
    macro_f1 = sum(v["f1"]        for v in per_class.values()) / max(len(labels), 1)
    macro_p  = sum(v["precision"] for v in per_class.values()) / max(len(labels), 1)
    macro_r  = sum(v["recall"]    for v in per_class.values()) / max(len(labels), 1)
    return {
        "accuracy":        round(accuracy, 4),
        "macro_f1":        round(macro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall":    round(macro_r, 4),
        "per_class":       per_class,
    }


def confusion_matrix(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, dict[str, int]]:
    cm: dict[str, dict[str, int]] = {
        t: {p: 0 for p in labels} for t in labels
    }
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm:
            cm[t][p] += 1
    return cm


def escalation_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Binary escalation quality (ESCALATE = positive class)."""
    tp = sum(t == "ESCALATE" and p == "ESCALATE" for t, p in zip(y_true, y_pred))
    fp = sum(t == "AUTO"     and p == "ESCALATE" for t, p in zip(y_true, y_pred))
    fn = sum(t == "ESCALATE" and p == "AUTO"     for t, p in zip(y_true, y_pred))
    tn = sum(t == "AUTO"     and p == "AUTO"     for t, p in zip(y_true, y_pred))
    n  = len(y_true)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    false_escalation_rate  = fp / max(tn + fp, 1)
    missed_escalation_rate = fn / max(tp + fn, 1)

    return {
        "accuracy":              round((tp + tn) / max(n, 1), 4),
        "precision":             round(prec, 4),
        "recall":                round(rec, 4),
        "f1":                    round(f1, 4),
        "false_escalation_rate": round(false_escalation_rate, 4),
        "missed_escalation_rate": round(missed_escalation_rate, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate() -> dict:
    with GOLDEN.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    labeled = [r for r in rows if r.get("intent") and r.get("expected_action")]
    if not labeled:
        raise ValueError(
            "No human labels found in golden_set.csv. "
            "Run src/label_golden.py first."
        )

    print(f"Evaluating AI agent on {len(labeled)} labeled golden-set rows...")
    print("(This may take 60-90 s as it runs retrieval for each message.)\n")

    result_rows: list[dict] = []
    y_true_intent: list[str] = []
    y_pred_intent: list[str] = []
    y_true_action: list[str] = []
    y_pred_action: list[str] = []

    for i, row in enumerate(labeled):
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(labeled)}")

        actual = agent_run(row["message"])

        y_true_intent.append(row["intent"])
        y_pred_intent.append(actual["intent"])
        y_true_action.append(row["expected_action"])
        y_pred_action.append(actual["action"])

        result_rows.append({
            "tweet_id":          row["tweet_id"],
            "message":           row["message"],
            "true_intent":       row["intent"],
            "predicted_intent":  actual["intent"],
            "intent_match":      int(row["intent"] == actual["intent"]),
            "true_action":       row["expected_action"],
            "predicted_action":  actual["action"],
            "action_match":      int(row["expected_action"] == actual["action"]),
            "confidence":        actual["confidence"],
            "triggered":         "|".join(actual.get("triggered", [])),
            "reply":             actual["reply"],
            "n_evidence":        len(actual.get("evidence", [])),
        })

    intent_labels = sorted(set(y_true_intent))
    action_labels = ["AUTO", "ESCALATE"]

    intent_metrics = compute_clf_metrics(y_true_intent, y_pred_intent, intent_labels)
    cm             = confusion_matrix(y_true_intent, y_pred_intent, intent_labels)
    esc_metrics    = escalation_metrics(y_true_action, y_pred_action)

    metrics = {
        "agent":              "keyword_jaccard_agent_v1",
        "n_samples":          len(labeled),
        "intent_metrics":     intent_metrics,
        "confusion_matrix":   cm,
        "escalation_metrics": esc_metrics,
    }

    # ─── Save results ──────────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    OUT_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # ─── Print summary ─────────────────────────────────────────────────────────
    print("\n=== AI Agent Evaluation Results ===\n")
    print(f"Intent Classification (n={len(labeled)}):")
    print(f"  Accuracy   : {intent_metrics['accuracy']:.4f}")
    print(f"  Macro F1   : {intent_metrics['macro_f1']:.4f}")
    print(f"  Macro Prec : {intent_metrics['macro_precision']:.4f}")
    print(f"  Macro Rec  : {intent_metrics['macro_recall']:.4f}")
    print(f"\n  Per-class F1:")
    for label, vals in intent_metrics["per_class"].items():
        print(f"    {label:30s}  F1={vals['f1']:.3f}  (n={vals['support']})")

    print(f"\nEscalation Decision:")
    print(f"  Accuracy         : {esc_metrics['accuracy']:.4f}")
    print(f"  Precision        : {esc_metrics['precision']:.4f}")
    print(f"  Recall           : {esc_metrics['recall']:.4f}")
    print(f"  F1               : {esc_metrics['f1']:.4f}")
    print(f"  False Esc. Rate  : {esc_metrics['false_escalation_rate']:.4f}")
    print(f"  Missed Esc. Rate : {esc_metrics['missed_escalation_rate']:.4f}")

    print(f"\nSaved -> {OUT_CSV}")
    print(f"Saved -> {OUT_JSON}")

    return metrics


if __name__ == "__main__":
    evaluate()
