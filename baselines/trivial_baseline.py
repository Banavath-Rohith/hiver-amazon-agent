"""Trivial baseline: always predicts the majority class from the golden set.

This baseline establishes the floor performance — any useful model must
beat it.  We compute the majority intent and majority action from the
labeled golden set so nothing is hard-coded.

Usage (from project root):
    python baselines/trivial_baseline.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

GOLDEN = Path(__file__).parent.parent / "data" / "golden" / "golden_set.csv"


def load_labeled(path: Path = GOLDEN) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    labeled = [r for r in rows if r.get("intent") and r.get("expected_action")]
    if not labeled:
        raise ValueError(
            "No labeled rows found. Run src/label_golden.py first to add labels."
        )
    return labeled


def compute_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict:
    """Accuracy, per-class precision/recall/F1, and macro averages."""
    n = len(y_true)
    if n == 0:
        return {}

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / n

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

    macro_p  = sum(v["precision"] for v in per_class.values()) / len(labels)
    macro_r  = sum(v["recall"]    for v in per_class.values()) / len(labels)
    macro_f1 = sum(v["f1"]        for v in per_class.values()) / len(labels)

    return {
        "accuracy":    round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall":    round(macro_r, 4),
        "macro_f1":        round(macro_f1, 4),
        "per_class":   per_class,
    }


def run() -> dict:
    rows = load_labeled()

    intent_counts = Counter(r["intent"] for r in rows)
    action_counts = Counter(r["expected_action"] for r in rows)

    majority_intent = intent_counts.most_common(1)[0][0]
    majority_action = action_counts.most_common(1)[0][0]

    print(f"Majority intent : {majority_intent}  ({intent_counts[majority_intent]}/{len(rows)})")
    print(f"Majority action : {majority_action}  ({action_counts[majority_action]}/{len(rows)})")

    y_true_intent = [r["intent"]          for r in rows]
    y_true_action = [r["expected_action"] for r in rows]
    y_pred_intent = [majority_intent]      * len(rows)
    y_pred_action = [majority_action]      * len(rows)

    intent_labels = sorted(set(y_true_intent))
    action_labels = sorted(set(y_true_action))

    intent_metrics = compute_metrics(y_true_intent, y_pred_intent, intent_labels)
    action_metrics = compute_metrics(y_true_action, y_pred_action, action_labels)

    result = {
        "baseline":       "trivial_majority",
        "majority_intent": majority_intent,
        "majority_action": majority_action,
        "n_samples":      len(rows),
        "intent":         intent_metrics,
        "action":         action_metrics,
    }

    print("\n-- Intent metrics ---------------------------")
    print(f"  Accuracy  : {intent_metrics['accuracy']:.4f}")
    print(f"  Macro F1  : {intent_metrics['macro_f1']:.4f}")
    print(f"  Macro P   : {intent_metrics['macro_precision']:.4f}")
    print(f"  Macro R   : {intent_metrics['macro_recall']:.4f}")

    print("\n-- Action metrics ---------------------------")
    print(f"  Accuracy  : {action_metrics['accuracy']:.4f}")
    print(f"  Macro F1  : {action_metrics['macro_f1']:.4f}")

    return result


if __name__ == "__main__":
    result = run()
    out_path = Path(__file__).parent.parent / "results" / "trivial_baseline_metrics.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")
