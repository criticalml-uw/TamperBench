#!/usr/bin/env python
"""Test curation on a small number of samples using the full curate_batch pipeline.

Usage:
    python scripts/ctrl/test_curation.py --num-samples 10
    python scripts/ctrl/test_curation.py --num-samples 20 --num-rounds 3
"""

import argparse
import tempfile
from pathlib import Path

import datasets
from dotenv import load_dotenv

from tamperbench.whitebox.defenses.ctrl.curation import (
    curate_batch,
    extract_query_response,
)
from tamperbench.whitebox.defenses.ctrl.types import Input


def test_curation(
    model_path: str,
    num_samples: int = 10,
    num_rounds: int = 2,
    beam_width: int = 2,
) -> None:
    """Test curation using the full curate_batch pipeline."""
    print(f"Testing curation with {num_samples} samples")
    print(f"Model: {model_path}")
    print(f"Rounds: {num_rounds}, Beam width: {beam_width}")
    print("=" * 80)

    # Load dataset
    raw_dataset: datasets.Dataset = datasets.load_dataset("tatsu-lab/alpaca", split="train")  # type: ignore[assignment]

    # Select diverse samples (not just first N)
    step = len(raw_dataset) // num_samples
    indices = [i * step for i in range(num_samples)]
    samples = [raw_dataset[i] for i in indices]

    # Extract query/response pairs
    inputs = [Input(*extract_query_response(s)) for s in samples]

    # Create temp cache directory
    with tempfile.TemporaryDirectory() as cache_dir:
        print(f"\nCache dir: {cache_dir}")
        print("\nRunning curate_batch...")

        # Run full curation pipeline
        curated_pairs = curate_batch(
            inputs=inputs,
            model_path=model_path,
            cache_dir=Path(cache_dir),
            num_rounds=num_rounds,
            temperatures=[0.4, 0.6],  # Reduced for faster testing
            top_ps=[0.6, 0.8],  # Reduced for faster testing
            beam_width=beam_width,
        )

    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    meta_patterns = [
        "improved response",
        "revised version",
        "here's a",
        "here is a",
        "rewritten",
    ]

    meta_count = 0
    truncated_count = 0
    same_count = 0

    for i, ((query, curated), inp) in enumerate(zip(curated_pairs, inputs, strict=False)):
        original = inp.response

        # Check for issues
        curated_lower = curated.lower()
        has_meta = any(p in curated_lower for p in meta_patterns)
        is_truncated = curated and curated[-1] not in ".!?\n\"')" and len(curated) > 50
        is_same = curated.strip() == original.strip()

        if has_meta:
            meta_count += 1
        if is_truncated:
            truncated_count += 1
        if is_same:
            same_count += 1

        print(f"\n--- Sample {i + 1} ---")
        print(f"QUERY: {query[:80]}{'...' if len(query) > 80 else ''}")
        print(f"ORIGINAL: {original[:120]}{'...' if len(original) > 120 else ''}")
        print(f"CURATED: {curated[:120]}{'...' if len(curated) > 120 else ''}")
        print(f"[meta={has_meta}, truncated={is_truncated}, same={is_same}]")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples: {num_samples}")
    print(f"With meta-commentary: {meta_count} ({meta_count / num_samples * 100:.1f}%)")
    print(f"Possibly truncated: {truncated_count} ({truncated_count / num_samples * 100:.1f}%)")
    print(f"Unchanged: {same_count} ({same_count / num_samples * 100:.1f}%)")

    # Quality check
    if meta_count / num_samples > 0.1:
        print("\n⚠️  WARNING: >10% responses have meta-commentary")
    if truncated_count / num_samples > 0.1:
        print("\n⚠️  WARNING: >10% responses may be truncated")
    if same_count / num_samples > 0.5:
        print("\n⚠️  WARNING: >50% responses unchanged (curation not effective)")
    if meta_count / num_samples <= 0.1 and truncated_count / num_samples <= 0.1 and same_count / num_samples <= 0.5:
        print("\n✓ Curation quality looks good!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test curation pipeline")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for curation",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--num-rounds", type=int, default=2, help="Number of beam search rounds")
    parser.add_argument("--beam-width", type=int, default=2, help="Beam width for search")
    args = parser.parse_args()

    load_dotenv()
    test_curation(
        args.model,
        args.num_samples,
        args.num_rounds,
        args.beam_width,
    )
