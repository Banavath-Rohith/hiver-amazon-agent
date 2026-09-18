"""Extract safely linked AmazonHelp customer/support pairs from TWCS."""
from __future__ import annotations
import csv
from pathlib import Path
RAW=Path("data/raw/twcs.csv"); OUT=Path("data/processed/amazon_cases.csv")
def main() -> None:
    # A pair is accepted only when an inbound tweet explicitly names an outbound
    # response tweet authored by AmazonHelp. This avoids proximity-based pairing.
    with RAW.open(encoding="utf-8",newline="") as f: rows={r["tweet_id"]:r for r in csv.DictReader(f) if r.get("tweet_id")}
    cases=[]
    for row in rows.values():
        if row.get("inbound","").lower() != "true": continue
        for response_id in row.get("response_tweet_id","").split(","):
            reply=rows.get(response_id.strip())
            if reply and reply.get("author_id")=="AmazonHelp" and reply.get("inbound","").lower()=="false":
                cases.append({"tweet_id":row["tweet_id"],"message":row.get("text","").strip(),"support_tweet_id":reply["tweet_id"],"support_response":reply.get("text","").strip()})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["tweet_id","message","support_tweet_id","support_response"]); writer.writeheader(); writer.writerows(cases)
    print(f"Wrote {len(cases)} explicit AmazonHelp conversation pairs to {OUT}")
if __name__ == "__main__": main()
