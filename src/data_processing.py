"""Reproducible preprocessing for the Customer Support on Twitter dataset.

The raw ``text`` column is copied unchanged.  This module adds a separate,
whitespace-normalized column for later analysis, so that every downstream
decision can still be traced back to the original tweet.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_INPUT = Path("data/raw/twcs.csv")
DEFAULT_OUTPUT = Path("data/processed/twcs_processed.csv")
REQUIRED_COLUMNS = {
	"tweet_id",
	"author_id",
	"inbound",
	"created_at",
	"text",
	"response_tweet_id",
	"in_response_to_tweet_id",
}


def normalize_text(text: str | None) -> str:
	"""Normalize whitespace while keeping the original message unchanged."""
	return re.sub(r"\s+", " ", text or "").strip()


def process_csv(
	input_path: Path = DEFAULT_INPUT,
	output_path: Path = DEFAULT_OUTPUT,
	max_rows: int | None = None,
) -> int:
	"""Validate and process the CSV, returning the number of written rows.

	``max_rows`` is an optional deterministic prefix limit for quick, local
	experiments.  It is deliberately opt-in: a future analysis should record the
	value it used rather than silently working on a sample.
	"""
	if not input_path.is_file():
		raise FileNotFoundError(
			f"Dataset not found at {input_path}. Place the downloaded twcs.csv there."
		)
	if max_rows is not None and max_rows < 1:
		raise ValueError("max_rows must be at least 1 when it is provided")
	if input_path.resolve() == output_path.resolve():
		raise ValueError("Input and output paths must be different")

	with input_path.open("r", encoding="utf-8-sig", newline="") as source:
		reader = csv.DictReader(source)
		columns = set(reader.fieldnames or [])
		missing_columns = REQUIRED_COLUMNS - columns
		if missing_columns:
			missing = ", ".join(sorted(missing_columns))
			raise ValueError(f"Unexpected dataset schema; missing columns: {missing}")

		output_path.parent.mkdir(parents=True, exist_ok=True)
		fieldnames = list(reader.fieldnames or []) + ["normalized_text"]
		row_count = 0
		with output_path.open("w", encoding="utf-8", newline="") as destination:
			writer = csv.DictWriter(destination, fieldnames=fieldnames)
			writer.writeheader()
			for row in reader:
				row["normalized_text"] = normalize_text(row["text"])
				writer.writerow(row)
				row_count += 1
				if max_rows is not None and row_count >= max_rows:
					break

	return row_count


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument(
		"--max-rows",
		type=int,
		default=None,
		help="Optionally process a deterministic prefix of the dataset.",
	)
	args = parser.parse_args()
	try:
		row_count = process_csv(args.input, args.output, args.max_rows)
	except (FileNotFoundError, ValueError) as error:
		parser.error(str(error))
	print(f"Processed {row_count} rows into {args.output}")


if __name__ == "__main__":
	main()
