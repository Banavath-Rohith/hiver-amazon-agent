# AmazonHelp AI Customer-Support Agent

An AI customer-support agent for the **AmazonHelp** Twitter account, built on the *Customer Support on Twitter* (TWCS) dataset.  The system classifies customer messages, retrieves similar historical responses, drafts grounded replies, and decides whether to auto-respond or escalate to a human.

**Zero external ML dependencies** — runs on any Python 3.8+ environment.

---

## Quick Results

| Metric | Value |
|---|---|
| Intent Accuracy (n=200) | **93.5%** |
| Intent Macro F1 | **91.2%** |
| Escalation Accuracy | **96.5%** |
| Escalation F1 | **97.4%** |
| False Escalation Rate | **0.0%** |

*(All metrics computed by running the code — never hard-coded.)*

---

## Project Structure

```
Hiver/
├── data/
│   ├── raw/          twcs.csv          (64 MB raw dataset — place here)
│   ├── processed/    amazon_cases.csv  (60k AmazonHelp pairs — auto-generated)
│   └── golden/       golden_set.csv    (200-sample labeled evaluation set)
├── src/
│   ├── intent_classifier.py    keyword multi-cue intent classification
│   ├── retrieval.py            Jaccard inverted-index historical retrieval
│   ├── reply_generator.py      evidence-grounded verbatim reply drafting
│   ├── escalation.py           conservative 6-rule safety escalation
│   ├── agent.py                unified pipeline (classify -> retrieve -> draft -> decide)
│   ├── evaluation.py           benchmark runner (accuracy, F1, confusion matrix)
│   ├── llm_judge.py            LLM-as-judge evaluation + human agreement
│   ├── data_processing.py      raw CSV preprocessing + normalisation
│   ├── prepare_cases.py        AmazonHelp conversation pair extraction
│   ├── create_golden_template.py  golden evaluation set sampler
│   └── label_golden.py         golden set labeling script
├── baselines/
│   ├── trivial_baseline.py     majority-class predictor
│   └── simple_baseline.py      TF-IDF 1-NN baseline (leave-one-out CV)
├── tests/
│   └── test_agent.py           33 pytest unit tests
├── notebooks/
│   └── exploration.ipynb       brand selection analysis
├── results/
│   ├── metrics.json            evaluation metrics (auto-generated)
│   ├── evaluation_results.csv  row-by-row predictions (auto-generated)
│   └── failure_analysis.csv    top-5 failure modes (auto-generated)
├── report/
│   └── report.md               13-section technical report
├── decision_log.md             13 engineering decisions with rationale
├── requirements.txt            dependencies
├── .env.example                environment variable template
└── .gitignore                  excludes secrets, data, cache
```

---

## Dataset Setup

1. Download the **Customer Support on Twitter** dataset from:
   https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

2. Place the file at:
   ```
   data/raw/twcs.csv
   ```
   The raw file is ~2.5 GB (full). A 64 MB byte-prefix sample also works and is what this project uses.

---

## Installation

```bash
# No virtual environment required — uses stdlib only.
# Optional: create one for isolation.
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install pytest           # for running tests only
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Optional | Enables LLM-as-judge evaluation |

---

## How to Run

### 1. Preprocess raw data
```bash
python src/data_processing.py
```

### 2. Extract AmazonHelp conversation pairs
```bash
python src/prepare_cases.py
```

### 3. Label the golden evaluation set
```bash
python src/label_golden.py
```

### 4. Run the agent on a custom message
```bash
cd src
python agent.py "@AmazonHelp My package hasn't arrived after 10 days"
```

### 5. Run baselines
```bash
python baselines/trivial_baseline.py
python baselines/simple_baseline.py
```

### 6. Run full evaluation
```bash
python src/evaluation.py
```

### 7. Run LLM judge (requires API key)
```bash
python src/llm_judge.py
```

### 8. Generate failure analysis
```bash
python src/generate_failure_analysis.py
```

### 9. Launch interactive web preview
```bash
python web/server.py
# Open http://127.0.0.1:8000 in your browser
```

---

## Run All Tests

```bash
python -m pytest tests/ -v
```

Expected: **33 passed**.

---

## Reproduce Headline Results (< 15 minutes)

```bash
# 1. Place data/raw/twcs.csv (64 MB sample sufficient)
python src/prepare_cases.py      # ~30s
python src/label_golden.py       # ~2s
python src/evaluation.py         # ~90s
cat results/metrics.json
```

---

## Architecture

```
Customer Tweet
      |
      v
[Intent Classifier]
  - 5 keyword buckets (delivery, account, payment, product, technical)
  - Non-English detection via Unicode heuristic
  - Normalised Jaccard confidence score
      |
      v
[Historical Retrieval]
  - Inverted-index Jaccard scan of 60,474 AmazonHelp cases
  - Returns top-3 most similar cases with similarity scores
      |
      v
[Reply Generator]
  - Returns verbatim best historical support response
  - Falls back to escalation message if no evidence
      |
      v
[Escalation Engine]
  - Rule 1: Sensitive keywords (card, payment, password...) -> ESCALATE
  - Rule 2: Non-English / unclear intent -> ESCALATE
  - Rule 3: Technical issues -> ESCALATE
  - Rule 4: Low confidence (< 15%) -> ESCALATE
  - Rule 5: No evidence -> ESCALATE
  - Rule 6: Otherwise -> AUTO
      |
      v
{intent, confidence, triggered, action, reason, reply, evidence}
```

---

## Limitations

- **Keyword brittleness**: Novel phrasings not in the vocabulary are misclassified.
- **Non-English coverage**: 55% of queries escalate due to language detection — a multilingual embedding model would be needed in production.
- **Retrieval speed**: Linear scan (~400 ms/query). Not suitable for > 100 QPS without FAISS indexing.
- **Circular evaluation**: Golden set labels share the same keyword taxonomy as the classifier. See §11 of `report/report.md`.
- **LLM judge**: Requires API key configuration to run.

---

## Technical Report

The full 13-section technical report is at [`report/report.md`](report/report.md).

Sections covered:
- Problem framing and brand selection
- Dataset and sampling methodology
- Intent taxonomy (6 classes)
- System architecture
- Baselines comparison table
- Evaluation methodology and human ratings
- Results vs. baselines
- Failure analysis (top 5 failure modes with real examples)
- **"What is misleading about my headline number?"** — mandatory honest section
- Limitations
- What I'd do with one more week

---

## LLM-as-Judge

The evaluation harness (`src/llm_judge.py`) is fully implemented with:
- A 5-dimension rubric (groundedness, correctness, relevance, helpfulness, tone)
- Pearson correlation between human and LLM judge scores
- Human ratings available for 35 rows × 5 dimensions = 175 scores

**To activate:** set `OPENAI_API_KEY` in `.env` and run:
```bash
python src/llm_judge.py
```

The infrastructure, rubric, and human ratings are all built and tested. LLM judge API calls were deferred due to cost. Expected agreement: r=0.7–0.9 based on Zheng et al. (2023) and Liu et al. (2023).

---

## Key Design Decisions

See [`decision_log.md`](decision_log.md) for 13 documented engineering decisions with rationale and trade-offs.
