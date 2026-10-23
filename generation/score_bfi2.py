"""Rescore a CSV containing Q1--Q60 without making model requests."""
from pathlib import Path
import argparse
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from generation.bfi2 import score_bfi2

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Choose a separate output file; original responses must be retained")
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [f"Q{i}" for i in range(1, 61) if f"Q{i}" not in fields]
        if missing:
            parser.error("Missing CSV columns: " + ", ".join(missing))
        rows = []
        for number, row in enumerate(reader, start=2):
            try:
                scores = score_bfi2(row)
            except ValueError as exc:
                raise ValueError(f"CSV row {number}: {exc}") from exc
            rows.append({**row, **{f"BFI2_{domain}": value for domain, value in scores.items()}})
    output_fields = fields + [f"BFI2_{d}" for d in ["E", "A", "C", "N", "O"] if f"BFI2_{d}" not in fields]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scored {len(rows)} complete records into {args.output}")

if __name__ == "__main__":
    main()
