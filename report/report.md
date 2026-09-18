# AmazonHelp AI Customer-Support Agent — Technical Report

## 1. Executive Summary

This report describes an AI customer-support agent built for the **AmazonHelp** Twitter account, trained on the *Customer Support on Twitter* (TWCS) dataset.  The agent classifies incoming customer messages into one of six intents, retrieves the most similar historical AmazonHelp responses, drafts a grounded reply, and decides whether to auto-respond or escalate to a human.

**Headline metric (on 200-sample labeled golden set):**

| Component | Metric | Value |
|---|---|---|
| Intent classifier | Accuracy | **93.5%** |
| Intent classifier | Macro F1 | **91.2%** |
| Escalation decision | Accuracy | **96.5%** |
| Escalation decision | F1 | **97.4%** |
| Escalation decision | False Escalation Rate | **0.0%** |

These numbers are real — computed by running the pipeline on every row of the golden evaluation set.  See §11 for why they may be optimistic.

---

## 2. Problem Framing

**Goal:** Given a raw customer tweet directed at a brand, an AI system should (a) understand what the customer needs, (b) draft a helpful, factually grounded reply, and (c) decide whether a human must intervene.

**Constraints chosen:**
- No external ML libraries (zero-dependency stdlib implementation) to maximise reproducibility across environments.
- No invented policy claims in replies (retrieved verbatim from real AmazonHelp responses).
- Conservative escalation to prevent automated harm on financial/security queries.

---

## 3. Dataset and Sampling

- **Source:** Customer Support on Twitter (TWCS), Kaggle / thoughtvector.
- **Raw file:** `data/raw/twcs.csv` — 64 MB byte-prefix sample (first ~288 k tweets).
- **Conversation extraction:** Only tweet pairs explicitly linked by `response_tweet_id` are accepted. This avoids false pairing of unrelated adjacent rows (Decision Log #3).
- **AmazonHelp pairs extracted:** 60,474 linked customer/support pairs (`data/processed/amazon_cases.csv`, 8.5 MB).
- **Golden evaluation set:** 200 rows sampled with `random.Random(42)` for reproducibility, then labeled by the systematic taxonomy below.

---

## 4. Selected Brand: AmazonHelp

AmazonHelp was selected because:
- Highest explicit inbound/outbound pair volume in the dataset.
- English-dominant (though ~55% of the golden set was non-English — a discovered limitation).
- Well-defined support categories (delivery, payment, account, product, technical).

---

## 5. Intent Taxonomy

Six intents derived from actual AmazonHelp conversation analysis:

| Intent | Description | Golden Set n |
|---|---|---|
| `delivery_or_order` | Package delays, tracking, missing items | 50 |
| `other_or_unclear` | Non-English, ambiguous, no keywords | 110 |
| `account_or_login` | Password, login, locked account | 10 |
| `payment_or_refund` | Charges, refunds, billing | 11 |
| `product_or_service` | Devices, subscriptions, item quality | 7 |
| `technical_issue` | App errors, website issues | 12 |

The dominance of `other_or_unclear` (55%) reflects the multilingual nature of the TWCS dataset — a critical real-world challenge (see §11).

---

## 6. System Architecture

```
Customer Tweet
      |
      v
[Intent Classifier]      -- keyword n-gram scoring, non-English heuristic
      |
      v
[Historical Retrieval]   -- Jaccard-similarity inverted-index over 60k cases
      |
      v
[Reply Generator]        -- returns best verbatim historical support response
      |
      v
[Escalation Engine]      -- 6-rule deterministic safety check
      |
      v
Structured JSON Output   {intent, confidence, action, reason, reply, evidence}
```

All components are in `src/` and are independently testable.

---

## 7. Baselines

### Trivial Baseline (always predict majority class)
- Majority intent: `other_or_unclear` (55% of golden set)
- Intent Accuracy: **55.0%** | Macro F1: **11.8%**
- Action Accuracy: **68.5%** | Macro F1: **40.7%**

### Simple Baseline (TF-IDF unigram+bigram 1-NN, leave-one-out CV)
- Intent Accuracy: **54.0%** | Macro F1: **27.6%**
- Action Accuracy: **65.5%** | Macro F1: **58.7%**

The TF-IDF baseline improves on macro F1 (better minority class recall) but not accuracy, showing that the majority-class effect dominates accuracy.

---

## 8. Evaluation Methodology

- **Golden set:** 200 rows sampled with `random.Random(42)` for reproducibility. Full sampling and labeling methodology documented in `data/golden/LABELING_NOTE.md`.
- **Labeling method:** Deterministic keyword rules (same vocabulary as classifier) assign intent labels; the same escalation rule set assigns action labels. This ensures consistency but creates a circular evaluation — disclosed in §11.
- **Human ratings:** First 35 rows rated 1–5 on five dimensions: groundedness, correctness, relevance, helpfulness, professional tone. Total: 175 human scores across 5 dimensions. Ratings are stored in `data/golden/golden_set.csv` columns `human_groundedness`, `human_correctness`, `human_relevance`, `human_helpfulness`, `human_tone`.
- **LLM judge (infrastructure built, API calls deferred):** The rubric, Pearson correlation function, and API integration are fully implemented in `src/llm_judge.py`. The judge evaluates 5 dimensions (groundedness, correctness, relevance, helpfulness, tone) on a 1–5 scale, calls GPT-4o-mini via OpenAI API, and computes per-dimension and overall Pearson correlation against the 35 human-rated rows. **LLM API calls were not executed** because this requires a paid `OPENAI_API_KEY`. The expected output format is demonstrated in `results/llm_judge_sample.json`. Based on published benchmarks (Zheng et al., 2023; Liu et al., 2023), GPT-4 class judges achieve r=0.7–0.9 human agreement on structured rubrics. To run: set `OPENAI_API_KEY` in `.env` and execute `python src/llm_judge.py`.
- **Escalation evaluation:** Binary metrics with ESCALATE as the positive class.
- **All intent/escalation metrics are computed by running the code** — no figures are manually typed.

---

## 9. Results

### AI Agent vs Baselines

| Metric | Trivial | TF-IDF 1-NN | AI Agent |
|---|---|---|---|
| Intent Accuracy | 55.0% | 54.0% | **93.5%** |
| Intent Macro F1 | 11.8% | 27.6% | **91.2%** |
| Escalation Accuracy | 68.5% | 65.5% | **96.5%** |
| Escalation F1 | 40.7% | 58.7% | **97.4%** |
| False Escalation Rate | — | — | **0.0%** |

### Per-Intent F1 (AI Agent)

| Intent | F1 | Support |
|---|---|---|
| `other_or_unclear` | 0.957 | 110 |
| `account_or_login` | 0.952 | 10 |
| `product_or_service` | 0.933 | 7 |
| `delivery_or_order` | 0.925 | 50 |
| `payment_or_refund` | 0.857 | 11 |
| `technical_issue` | 0.846 | 12 |

---

## 10. Failure Analysis

From 200 predictions, 13 errors were found. Top 5 failure modes (see `results/failure_analysis.csv`):

1. **Missed delivery keywords** — Message describes a delivery problem using natural language without triggering vocabulary terms.
2. **Payment/delivery confusion** — Keyword overlap when a message mentions 'cancel' and 'order' activates both buckets.
3. **Missed escalation** — Subtle situations requiring human judgement (e.g., complex international orders) don't trigger keyword rules.
4. **Technical issue indirect phrasing** — "it just keeps spinning" doesn't match any tech keywords.
5. **Mixed-language false activation** — English product names (Kindle, Prime) in non-English text trigger wrong intents.

---

## 11. What Is Misleading About My Headline Number?

**93.5% intent accuracy is optimistic for the following reasons:**

1. **Circular labeling**: The golden set was labeled using the same keyword taxonomy the classifier uses. A message is labeled `delivery_or_order` because it contains the word "order" — and the classifier also fires on "order". The evaluation tests consistency, not generalisation to truly unseen labeling standards.

2. **Class imbalance**: 55% of examples are `other_or_unclear`. A classifier that identifies these reliably (which keyword matching does via non-English detection) earns 55 free correct predictions. The minority classes (7 product, 10 account, 11 payment, 12 technical) drive quality but are statistically unreliable with n < 30.

3. **No train/test split**: The agent uses the full 60k-case retrieval database at eval time. In production, some queries will be genuinely novel with no historical matches.

4. **No noise robustness test**: Twitter messages include slang, typos, emojis, and sarcasm not well-represented in the golden set.

5. **LLM judge deferred**: Reply quality (groundedness, helpfulness, tone) has been assessed with human ratings on only 35 rows, without LLM cross-validation.

6. **Offline ≠ production**: A real production deployment would face distribution shift from new product launches, seasonal spikes, and evolving customer language.

---

## 12. Limitations

- **Keyword brittleness**: Novel phrasing not in the keyword vocabulary is misclassified.
- **Non-English coverage**: 55% of the TWCS AmazonHelp dataset is non-English; these are currently escalated by design but not resolved.
- **No intent generalisation**: The classifier cannot reason about new issue types not in the taxonomy.
- **Reply grounding**: Replies are verbatim historical responses; they may not perfectly address the new message even when the intent matches.
- **Retrieval speed**: The Jaccard retrieval scans all 60k cases linearly (~0.4 s/query). Not suitable for high-traffic production without pre-indexing.

---

## 13. What I Would Do With One More Week

1. **Embedding-based retrieval**: Replace Jaccard with sentence-transformer embeddings + FAISS for semantic matching across language barriers.
2. **Multilingual support**: Detect language and route to a language-specific retrieval index.
3. **Intent taxonomy expansion**: Add `complaint`, `compliment`, `general_inquiry` after deeper cluster analysis.
4. **LLM-generated replies**: Use retrieved evidence as few-shot context for an LLM reply that adapts tone and content to the specific query.
5. **Active learning**: Build a UI for human agents to correct predictions, feeding back into the retrieval database.
6. **Production stress test**: Benchmark retrieval latency at 1000 QPS and profile memory footprint.
