#!/usr/bin/env python3
"""Check uploaded HuggingFace dataset.

Usage:
    uv run scripts/strong_reject_rubric/check_dataset.py \
        --repo-id username/strongreject_with_targets
"""

import argparse
import os

import datasets
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    """Run the main script."""
    parser = argparse.ArgumentParser(description="Check HuggingFace dataset")

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., username/dataset-name)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to display (default: 5)",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.repo_id}")
    dataset = datasets.load_dataset(args.repo_id, split="train", token=os.getenv("HF_TOKEN"))

    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Total rows: {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")

    # Show samples
    print(f"\n--- First {min(args.num_samples, len(dataset))} samples ---\n")
    for i in range(min(args.num_samples, len(dataset))):
        print(f"Sample {i + 1}:")
        print(f"  Prompt: {dataset[i]['prompt'][:150]}...")
        print(f"  Response: {dataset[i]['target'][:150]}...")
        print()


if __name__ == "__main__":
    main()
