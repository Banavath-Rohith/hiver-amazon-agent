# Decision log
1. Used a 64 MB fixed byte-prefix sample rather than the full dataset; this is fast but ordering may bias brands.
2. Selected AmazonHelp from measured outbound volume and explicit linked-pair volume.
3. Accept only tweet-ID response links, not adjacent rows, to avoid false conversations.
4. Preserve raw text and use separate normalization.
5. Use transparent keyword/similarity intent logic for live explainability.
6. Retrieve historical replies verbatim rather than generate unsupported policy claims.
7. Escalate sensitive account/payment/security wording conservatively; these are design rules, not brand policy.
8. Golden rows are sampled from real cases but remain unlabeled until human review.
9. Evaluation refuses to run without human intent/action labels.
10. LLM judging is deferred until an API key and non-leaking judge prompt are configured.
