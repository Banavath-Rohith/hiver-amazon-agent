"""Simple classical NLP baseline: TF-IDF n-gram vectoriser + cosine similarity.

Why no scikit-learn?
--------------------
Python 3.14 ships no compatible wheel for scikit-learn at the time of this
project.  The TF-IDF + cosine similarity below is implemented from scratch
using only the standard library + math module so that:
  * the project runs on any Python 3.8+ environment with zero pip steps, and
  * reviewers can read and understand every calculation.

Approach
--------
* Build TF-IDF vectors over unigrams and bigrams of all training messages.
* At inference time compute cosine similarity to every training vector and
  predict the intent/action of the nearest neighbour (1-NN retrieval).

Usage (from project root):
    python baselines/simple_baseline.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

GOLDEN = Path(__file__).parent.parent / "data" / "golden" / "golden_set.csv"


# --- Text utilities ------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lower-case unigrams + bigrams."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return grams


# --- TF-IDF from scratch -------------------------------------------------------

def build_tfidf(docs: list[list[str]]) -> tuple[dict[str, int], list[dict[str, float]]]:
    """Return (vocab index, list of TF-IDF vectors as sparse dicts)."""
    N = len(docs)
    df: dict[str, int] = defaultdict(int)
    for doc in docs:
        for term in set(doc):
            df[term] += 1

    vocab = {term: idx for idx, term in enumerate(sorted(df))}
    idf   = {term: math.log((N + 1) / (cnt + 1)) + 1 for term, cnt in df.items()}

    vectors: list[dict[str, float]] = []
    for doc in docs:
        tf: dict[str, int] = defaultdict(int)
        for term in doc:
            tf[term] += 1
        vec = {term: (count / len(doc)) * idf[term]
               for term, count in tf.items() if term in idf}
        vectors.append(vec)

    return vocab, vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common_keys = a.keys() & b.keys()
    dot  = sum(a[k] * b[k] for k in common_keys)
    na   = math.sqrt(sum(v * v for v in a.values()))
    nb   = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def transform(tokens_list: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Convert a token list to a TF-IDF vector (sparse dict)."""
    if not tokens_list:
        return {}
    tf: dict[str, int] = defaultdict(int)
    for t in tokens_list:
        tf[t] += 1
    return {term: (cnt / len(tokens_list)) * idf.get(term, 0)
            for term, cnt in tf.items() if idf.get(term, 0) > 0}


# --- Metrics ------------------------------------------------------------------

def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
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
    macro_f1 = sum(v["f1"] for v in per_class.values()) / max(len(labels), 1)
    macro_p  = sum(v["precision"] for v in per_class.values()) / max(len(labels), 1)
    macro_r  = sum(v["recall"]    for v in per_class.values()) / max(len(labels), 1)
    return {
        "accuracy":        round(accuracy, 4),
        "macro_f1":        round(macro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall":    round(macro_r, 4),
        "per_class":       per_class,
    }


# --- Main ---------------------------------------------------------------------

def run() -> dict:
    with GOLDEN.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    labeled = [r for r in rows if r.get("intent") and r.get("expected_action")]
    if not labeled:
        raise ValueError("No labeled rows. Run src/label_golden.py first.")

    print(f"Training + evaluating 1-NN TF-IDF on {len(labeled)} labeled rows "
          f"(leave-one-out cross-validation)...")

    docs     = [tokenize(r["message"]) for r in labeled]
    intents  = [r["intent"]          for r in labeled]
    actions  = [r["expected_action"] for r in labeled]

    # Build IDF from the full corpus
    N = len(docs)
    df: dict[str, int] = defaultdict(int)
    for doc in docs:
        for term in set(doc):
            df[term] += 1
    idf = {term: math.log((N + 1) / (cnt + 1)) + 1 for term, cnt in df.items()}

    # Build TF-IDF vectors
    vectors: list[dict[str, float]] = [transform(doc, idf) for doc in docs]

    # Leave-one-out 1-NN prediction
    pred_intents: list[str] = []
    pred_actions: list[str] = []

    for i in range(N):
        q_vec = vectors[i]
        best_sim, best_j = -1.0, -1
        for j in range(N):
            if i == j:
                continue
            sim = cosine(q_vec, vectors[j])
            if sim > best_sim:
                best_sim, best_j = sim, j
        pred_intents.append(intents[best_j] if best_j >= 0 else "other_or_unclear")
        pred_actions.append(actions[best_j]  if best_j >= 0 else "ESCALATE")

    intent_labels = sorted(set(intents))
    action_labels = sorted(set(actions))

    intent_metrics = compute_metrics(intents, pred_intents, intent_labels)
    action_metrics = compute_metrics(actions, pred_actions, action_labels)

    result = {
        "baseline": "simple_tfidf_1nn_loocv",
        "n_samples": N,
        "intent":   intent_metrics,
        "action":   action_metrics,
    }

    print(f"\n-- Intent metrics (leave-one-out) ---------------------------")
    print(f"  Accuracy  : {intent_metrics['accuracy']:.4f}")
    print(f"  Macro F1  : {intent_metrics['macro_f1']:.4f}")
    print(f"  Macro P   : {intent_metrics['macro_precision']:.4f}")
    print(f"  Macro R   : {intent_metrics['macro_recall']:.4f}")
    print(f"\n  Per-class F1:")
    for label, vals in intent_metrics["per_class"].items():
        print(f"    {label:30s}  F1={vals['f1']:.3f}  (n={vals['support']})")

    print(f"\n-- Action metrics ---------------------------")
    print(f"  Accuracy  : {action_metrics['accuracy']:.4f}")
    print(f"  Macro F1  : {action_metrics['macro_f1']:.4f}")

    return result


if __name__ == "__main__":
    result = run()
    out_path = Path(__file__).parent.parent / "results" / "simple_baseline_metrics.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")
