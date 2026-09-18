# Golden Evaluation Set — Sampling & Labeling Note

## Overview

**File:** `data/golden/golden_set.csv`  
**Size:** 200 examples (within the required 150–250 range)  
**Source:** Real AmazonHelp conversations from the TWCS dataset  
**Labeled by:** Systematic deterministic rules + manual review  

---

## How Examples Were Sampled

### Step 1 — Conversation Extraction
From `data/raw/twcs.csv` (~288k tweets, 64 MB byte-prefix sample), we extracted **60,474 explicit AmazonHelp conversation pairs** using `src/prepare_cases.py`.

**Pairing rule:** A pair is accepted only when a customer tweet's `response_tweet_id` field directly names an AmazonHelp outbound tweet. This avoids proximity-based pairing of unrelated messages.

### Step 2 — Stratified Random Sampling
From the 60,474 pairs, we sampled **200 examples** using:
```python
random.Random(seed=42).sample(rows, 200)
```
- `seed=42` makes the sample fully **reproducible**
- The sample is a uniform random draw — no stratification by intent was applied at sampling time

**Effect on distribution:** The 200 drawn examples reflect the natural distribution of AmazonHelp conversations. A large proportion (~55%) turned out to be non-English (French, Japanese, Portuguese) — a real-world finding, not cherry-picking.

---

## How Examples Were Labeled

### Label Schema

| Column | Values | Assigned by |
|---|---|---|
| `intent` | 6 canonical labels (see below) | Keyword taxonomy (same rules as classifier) |
| `expected_action` | `AUTO` / `ESCALATE` | Escalation rule set (same rules as agent) |
| `expected_reason` | Free text explanation | Derived from the rule that fired |
| `human_groundedness` | 1–5 | Manual review of 35 rows |
| `human_correctness` | 1–5 | Manual review of 35 rows |
| `human_relevance` | 1–5 | Manual review of 35 rows |
| `human_helpfulness` | 1–5 | Manual review of 35 rows |
| `human_tone` | 1–5 | Manual review of 35 rows |

### Intent Labels (6 Classes)

| Label | Keyword signals | Support (n) |
|---|---|---|
| `delivery_or_order` | order, delivery, tracking, arrived, delayed, shipment | 50 |
| `other_or_unclear` | Non-English OR no keyword match | 110 |
| `technical_issue` | app, website, error, crash, bug, not working | 12 |
| `payment_or_refund` | refund, charge, payment, card, billing, cancel | 11 |
| `account_or_login` | account, login, password, access, suspended | 10 |
| `product_or_service` | product, Kindle, Prime, item, subscribe, device | 7 |

### Labeling Method: Deterministic Keyword Rules

Labels were assigned by **the same deterministic keyword-matching rules that power the classifier at runtime** (`src/label_golden.py`). This approach was chosen because:

1. **Consistency** — manual labeling by a single reviewer introduces fatigue-related inconsistency. Rules apply identically to every row.
2. **Speed** — 200 rows labeled in < 2 seconds, allowing rapid iteration.
3. **Auditability** — every label can be traced to a specific keyword match or rule.

**Acknowledged limitation:** This creates a **circular evaluation** — the classifier and labels share vocabulary. The evaluation measures consistency within the taxonomy, not true generalisation to unseen labeling criteria. This is explicitly disclosed in Section 11 of `report/report.md` ("What Is Misleading About My Headline Number?").

### Escalation Label Rules

Labels follow the same priority chain as `src/escalation.py`:

1. Sensitive keywords present → `ESCALATE` (reason: account/payment/security keywords)
2. Intent = `other_or_unclear` → `ESCALATE` (reason: language or ambiguity barrier)
3. Intent = `technical_issue` → `ESCALATE` (reason: requires system access)
4. Confidence < 15% → `ESCALATE` (reason: uncertain classification)
5. No historical evidence → `ESCALATE` (reason: cannot ground reply)
6. Otherwise → `AUTO`

**Result:** 137 / 200 examples are `ESCALATE`, 63 / 200 are `AUTO`. The imbalance reflects real AmazonHelp risk profile — the majority of raw multilingual/ambiguous tweets require human review.

### Human Ratings (35-Row Subset)

The first 35 rows of the golden set have manual 1–5 ratings on five dimensions:

| Dimension | What 5 means | What 1 means |
|---|---|---|
| Groundedness | Reply stays within retrieved evidence | Reply contradicts or ignores evidence |
| Correctness | Information in reply is factually accurate | Contains incorrect information |
| Relevance | Reply addresses the customer's actual question | Completely off-topic |
| Helpfulness | Customer's issue would likely be resolved | Customer would be no better off |
| Professional Tone | Polite, empathetic, brand-appropriate | Rude, robotic, or off-brand |

Ratings reflect how well the **historical AmazonHelp response** in `historical_response` column addresses the customer message. These ratings are used for:
- LLM-as-judge agreement validation (`src/llm_judge.py`)
- Computing Pearson correlation between human and LLM judge scores

---

## Quality Checks Performed

1. ✅ All 200 rows have `intent` and `expected_action` labels
2. ✅ All 200 rows have `expected_reason` text
3. ✅ 35 rows have complete human ratings (1–5, all 5 dimensions)
4. ✅ No duplicate `tweet_id` values
5. ✅ No empty `message` fields
6. ✅ Intent distribution verified against raw dataset patterns
7. ✅ Escalation labels verified manually on a 20-row sample

---

## Reproducibility

To regenerate this exact golden set from scratch:
```bash
# 1. Re-extract AmazonHelp cases
python src/prepare_cases.py

# 2. Sample 200 rows (same seed=42)
python src/create_golden_template.py

# 3. Apply labels
python src/label_golden.py
```

The output will be byte-for-byte identical because `random.Random(42)` is deterministic.
