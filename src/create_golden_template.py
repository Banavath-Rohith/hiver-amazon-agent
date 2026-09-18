"""Create an unlabeled human-rating worksheet from actual AmazonHelp cases."""
from __future__ import annotations
import csv, random
from pathlib import Path
SOURCE=Path("data/processed/amazon_cases.csv"); OUTPUT=Path("data/golden/golden_set.csv")
def main(seed: int=42, size: int=200):
    with SOURCE.open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    chosen=random.Random(seed).sample(rows,min(size,len(rows)))
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open("w",encoding="utf-8",newline="") as f:
        fields=["tweet_id","message","historical_response","intent","expected_action","expected_reason","human_groundedness","human_correctness","human_relevance","human_helpfulness","human_tone"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in chosen: w.writerow({"tweet_id":r["tweet_id"],"message":r["message"],"historical_response":r["support_response"]})
    print(f"Created {len(chosen)} unlabeled rows at {OUTPUT}; a human must fill labels before evaluation.")
if __name__=="__main__": main()
