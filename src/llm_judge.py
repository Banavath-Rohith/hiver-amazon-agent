"""LLM-as-judge evaluation for AmazonHelp agent replies.

IMPORTANT: This module requires an OpenAI-compatible API key to be set in
the environment variable OPENAI_API_KEY (or equivalent).  Without an API
key the judge will NOT run.  Set one in .env (see .env.example).

Human vs LLM agreement
-----------------------
The golden set includes human ratings (1-5) for the first 35 rows across
five dimensions:
  groundedness, correctness, relevance, helpfulness, tone

This module:
  1. Calls the LLM judge for the same 35 rows.
  2. Computes Pearson correlation between human and LLM scores.
  3. Reports per-dimension agreement.

The rubric below is designed to minimize evaluation leakage:
  the judge sees the CUSTOMER MESSAGE and RETRIEVED EVIDENCE only,
  not the ground-truth label, so it cannot copy the answer.

Usage:
    OPENAI_API_KEY=sk-... python src/llm_judge.py
    (or set OPENAI_API_KEY in .env and use python-dotenv)
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from agent import run as agent_run

GOLDEN   = ROOT / "data" / "golden" / "golden_set.csv"
OUT_JSON = ROOT / "results" / "llm_judge_results.json"

# ─── Rubric ───────────────────────────────────────────────────────────────────
RUBRIC = """
You are a quality evaluator for an AI customer-support system.
Score the GENERATED REPLY on the following five dimensions, each 1-5:

1. Groundedness   – Does the reply stay within what the EVIDENCE supports?
                    (5 = fully grounded; 1 = contradicts or invents facts)
2. Correctness    – Is the information in the reply factually accurate
                    given the evidence?
3. Relevance      – Does the reply address the customer's actual question?
4. Helpfulness    – Would this reply likely satisfy the customer's need?
5. Professional Tone – Is the reply polite, professional, and empathetic?

CUSTOMER MESSAGE:
{message}

RETRIEVED EVIDENCE (historical responses the agent was shown):
{evidence}

GENERATED REPLY:
{reply}

Respond ONLY with a JSON object, for example:
{{"groundedness": 4, "correctness": 4, "relevance": 3, "helpfulness": 3, "tone": 4}}
"""


def _call_llm(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """Call the OpenAI Chat Completions API and return the raw reply text."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 100,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    num   = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return round(num / (den_x * den_y), 4)


def run_judge() -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-YOUR"):
        print(
            "OPENAI_API_KEY is not set or is a placeholder.\n"
            "Set it in your environment or in a .env file, then re-run.\n"
            "LLM judge evaluation is DEFERRED."
        )
        return {"status": "deferred", "reason": "No API key configured."}

    with GOLDEN.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Only use the 35 rows that have human ratings
    rated = [
        r for r in rows
        if r.get("intent") and r.get("human_groundedness")
    ][:35]

    print(f"Running LLM judge on {len(rated)} human-rated rows...")

    dims = ["groundedness", "correctness", "relevance", "helpfulness", "tone"]
    human_cols = [f"human_{d}" for d in dims]

    llm_scores: list[dict] = []
    for i, row in enumerate(rated):
        result = agent_run(row["message"])
        evidence_text = "\n".join(
            f"  [{j+1}] {e.get('support_response','')}"
            for j, e in enumerate(result.get("evidence", []))
        ) or "  (no evidence retrieved)"

        prompt = RUBRIC.format(
            message=row["message"],
            evidence=evidence_text,
            reply=result["reply"],
        )

        try:
            raw = _call_llm(prompt, api_key)
            scores = json.loads(raw)
        except Exception as exc:
            print(f"  Row {i}: judge error – {exc}")
            scores = {d: None for d in dims}

        llm_scores.append(scores)
        print(f"  {i+1}/{len(rated)}: {scores}")

    # ─── Compute agreement ─────────────────────────────────────────────────────
    agreement: dict[str, float] = {}
    for dim, hcol in zip(dims, human_cols):
        human_vals = []
        llm_vals   = []
        for row, llm in zip(rated, llm_scores):
            h = row.get(hcol)
            l = llm.get(dim)
            if h and l is not None:
                human_vals.append(float(h))
                llm_vals.append(float(l))
        agreement[dim] = pearson(human_vals, llm_vals) if len(human_vals) >= 2 else float("nan")

    overall_pairs: list[tuple[float, float]] = []
    for row, llm in zip(rated, llm_scores):
        for dim, hcol in zip(dims, human_cols):
            h = row.get(hcol)
            l = llm.get(dim)
            if h and l is not None:
                overall_pairs.append((float(h), float(l)))

    h_all, l_all = zip(*overall_pairs) if overall_pairs else ([], [])
    overall_r = pearson(list(h_all), list(l_all))

    result_data = {
        "status":           "completed",
        "n_rated_rows":     len(rated),
        "per_dim_pearson_r": agreement,
        "overall_pearson_r": overall_r,
        "llm_scores":       llm_scores,
    }

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(result_data, indent=2, ensure_ascii=False))
    print(f"\nOverall Pearson r (human vs LLM judge): {overall_r}")
    print(f"Per-dimension agreement: {agreement}")
    print(f"Saved -> {OUT_JSON}")
    return result_data


if __name__ == "__main__":
    # Support loading from .env file if present
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    run_judge()
