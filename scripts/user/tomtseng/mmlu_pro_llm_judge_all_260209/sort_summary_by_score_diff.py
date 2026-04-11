"""Sort summary.csv by the difference between judge and regex MMLU-Pro scores.

Usage:
    uv run python sort_summary_by_score_diff.py --summary /path/to/summary.csv --output /path/to/sorted.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    """Read summary CSV, sort by judge-regex score difference, and write to output."""
    parser = argparse.ArgumentParser(description="Sort summary.csv by judge_mmlu_pro_score - regex_mmlu_pro_score")
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write sorted output CSV",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with open(args.summary) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Sort by (judge - regex) ascending
    rows.sort(key=lambda r: float(r["judge_mmlu_pro_score"]) - float(r["regex_mmlu_pro_score"]))

    fieldnames = [
        "model",
        "attack",
        "trial",
        "regex_mmlu_pro_score",
        "judge_mmlu_pro_score",
        "diff",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            diff = float(r["judge_mmlu_pro_score"]) - float(r["regex_mmlu_pro_score"])
            writer.writerow(
                {
                    "model": r["model"],
                    "attack": r["attack"],
                    "trial": r["trial"],
                    "regex_mmlu_pro_score": r["regex_mmlu_pro_score"],
                    "judge_mmlu_pro_score": r["judge_mmlu_pro_score"],
                    "diff": f"{diff:+.6f}",
                }
            )

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
