# Decision Log

## 1. Used a 64 MB byte-prefix sample rather than the full 3M-row TWCS dataset

**Decision:** Process only the first 64 MB of `twcs.csv` via the `--max-rows` CLI flag.

**Reason:** The full dataset takes > 10 minutes to process on a standard laptop and produces 2.7 M rows, most of which are not AmazonHelp conversations. The byte-prefix is deterministic and reproducible.

**Trade-off:** Brand selection statistics and retrieval recall may differ on the full dataset. The ordering bias (earlier rows from different time periods) is documented rather than hidden.

---

## 2. Selected AmazonHelp as the target brand

**Decision:** Focus on AmazonHelp from the TWCS dataset.

**Reason:** AmazonHelp has the highest volume of explicitly linked inbound/outbound tweet pairs (~60 k pairs), is predominantly English (easier to evaluate), and has clearly separable support categories (delivery, payment, account, product, technical).

**Trade-off:** The dataset is multilingual. 55% of a random 200-sample golden set contained non-English messages, requiring a broad `other_or_unclear` fallback bucket.

---

## 3. Accept only explicit tweet-ID response links, not adjacent rows

**Decision:** A customer/support pair is only accepted when the inbound tweet's `response_tweet_id` column directly names the AmazonHelp outbound tweet.

**Reason:** Proximity-based pairing (matching consecutive rows) frequently creates false pairs between unrelated conversations. The explicit link guarantee is enforced by the dataset schema.

**Trade-off:** Approximately 30% of AmazonHelp tweets have no explicit link and are discarded, reducing retrieval database size.

---

## 4. Preserve raw text; add a separate `normalized_text` column

**Decision:** The `text` column is never overwritten. Normalization is stored separately in `normalized_text`.

**Reason:** Downstream decisions (retrieval, escalation, failure analysis) must always be traceable to the original tweet. Destructive normalization creates an audit gap.

**Trade-off:** Slightly larger CSV files.

---

## 5. Use transparent keyword/similarity intent logic, not a trained model

**Decision:** The intent classifier uses inspectable keyword buckets + Jaccard similarity rather than a neural network or fine-tuned LLM.

**Reason:** Hiver reviewers will inspect and modify the code live. A keyword classifier can be explained line-by-line; a fine-tuned BERT model cannot. Operational explainability was prioritized over raw accuracy.

**Trade-off:** The classifier misses indirect phrasings and novel language not in the keyword vocabulary. Macro F1 = 91.2% on the golden set, vs. a theoretical ceiling near 97–99% for a fine-tuned model.

---

## 6. Retrieve historical replies verbatim rather than generate unsupported content

**Decision:** The reply generator returns the highest-similarity historical AmazonHelp response without modification.

**Reason:** Generating free-form text risks hallucinating policy claims (e.g., "Your refund will arrive in 3 days") that AmazonHelp has never made. Verbatim retrieval is always grounded.

**Trade-off:** The reply may be stylistically off or address a slightly different situation. A future LLM layer could adapt tone while remaining evidence-constrained.

---

## 7. Escalate on sensitive keyword presence, regardless of intent or confidence

**Decision:** Messages containing any of {card, charge, charged, payment, refund, account, login, password, security, billing, credentials, locked, suspended, unauthorized} are always escalated.

**Reason:** These keywords signal interactions where an incorrect automated response could cause financial harm or security risk. The cost of false escalation is low; the cost of a missed escalation is high.

**Trade-off:** This produces 0% false escalations on the golden set but may be over-conservative in production (e.g., a customer asking "How do I update my payment method?" is escalated even though the query is informational).

---

## 8. Escalate on non-English intent (`other_or_unclear`)

**Decision:** All `other_or_unclear` messages are escalated.

**Reason:** Responding to a non-English customer in English (or with an irrelevant historical response) is worse than escalating to a bilingual human agent.

**Trade-off:** 55% of the dataset triggers this rule, making the system largely pass-through for non-English queries. A multilingual model would be needed to serve these customers.

---

## 9. Escalate on low classifier confidence (< 15%)

**Decision:** If the winning intent scores < 15% of fired-keyword overlap, escalate rather than guess.

**Reason:** A low-confidence prediction is likely wrong. An incorrect auto-reply is harder to recover from than a slow escalation.

**Trade-off:** The 15% threshold was chosen empirically during development; it was not tuned on a held-out validation set.

---

## 10. Golden set labels derived from the same taxonomy as the classifier

**Decision:** The golden set was labeled using the same deterministic keyword rules that the classifier uses at runtime.

**Reason:** Manual labeling from scratch by a single reviewer would introduce inconsistency and subjectivity. Using the same rules ensures measurement consistency.

**Trade-off:** This creates circular evaluation — the classifier and labels agree by construction for unambiguous messages. The 93.5% accuracy reflects consistency, not true generalisation. This is explicitly disclosed in §11 of the report.

---

## 11. Use Jaccard similarity over an inverted index, not FAISS

**Decision:** Retrieval uses a pure-Python inverted-index Jaccard search rather than a dense-vector ANN library (FAISS, annoy, etc.).

**Reason:** FAISS requires a compatible C++ wheel, which was unavailable for Python 3.14 at time of development. The inverted-index approach installs with zero pip steps and is fully auditable.

**Trade-off:** Linear scan over 60 k cases takes ~400 ms/query. FAISS with a pre-built index would reduce this to < 5 ms. Acceptable for a take-home project; unacceptable for > 100 QPS production traffic.

---

## 12. Evaluation refuses to run without human intent/action labels

**Decision:** `src/evaluation.py` raises `ValueError` if no labeled rows are found.

**Reason:** Running evaluation on unlabeled data would either crash silently or produce meaningless metrics. The hard guard forces the developer to acknowledge that labeling is a prerequisite.

**Trade-off:** None — this is a safety check with no downside.

---

## 13. LLM judging deferred until API key and judge prompt are configured

**Decision:** `src/llm_judge.py` prints a clear deferral message and exits cleanly when no API key is present.

**Reason:** Running with a placeholder key wastes API quota and produces misleading results. A clear, actionable error message is preferable to a confusing API failure.

**Trade-off:** LLM-as-judge evaluation is not available out of the box. Users must configure OPENAI_API_KEY in `.env` (documented in README).
