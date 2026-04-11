#!/usr/bin/env python3
"""Computes MMLU-Pro accuracy from TAR model's inferences.parquet files.

Outputs accuracies when parsing the answer from model responses using the following parsers:
- Regex parser
- LLM-as-a-judge parsing the letter answer from seeing only the response
- LLM-as-a-judge with full context, parsing the answer from seeing the response
  as well as the question and choices (to catch cases where the model doesn't
  follow the instructions to output the letter answer, but still gives the right
  answer in words)
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any

import datasets
import polars as pl

from tamperbench.utils import get_repo_root

# Add repo root to path so we can import from scripts.*
# We do this because scripts/ isn't in the distributed package.
sys.path.insert(0, str(get_repo_root()))

from scripts.user.tomtseng.mmlu_pro_llm_judge_251105.judge import (
    extract_with_retries,
)
from scripts.user.tomtseng.mmlu_pro_llm_judge_251105.run_judge import (
    strip_extra_questions_from_response,
)
from scripts.user.tomtseng.mmlu_pro_tar_251203.judge_with_context import (
    grade_with_context,
)
from tamperbench.whitebox.evals.mmlu_pro.eval_from_api import extract_answer, preprocess
from tamperbench.whitebox.evals.mmlu_pro.mmlu_pro import (
    DATASET_NAME,
    VAL_MAX_EXAMPLES_PER_SUBJECT,
    limit_examples_per_subject,
)


def extract_pred(response: str) -> str:
    """Extract prediction from PRED: at end of response."""
    match = re.search(r"PRED:\s*([A-P])", response)
    if match:
        return match.group(1)
    # Fallback to existing extraction logic
    return extract_answer(response) or ""


def strip_pred_suffix(response: str) -> str:
    """Strip the PRED: <letter or blank> suffix from response."""
    return re.sub(r"\s*PRED:\s*[A-P]?\s*$", "", response)


def build_ground_truth() -> list[dict[str, Any]]:
    """Load and preprocess MMLU-Pro test split, same as tamperbench."""
    ds = datasets.load_dataset(DATASET_NAME, split="test")
    # HuggingFace datasets library doesn't properly type Dataset iteration
    rows = [dict(item) for item in ds]
    rows = preprocess(rows)
    return limit_examples_per_subject(rows, VAL_MAX_EXAMPLES_PER_SUBJECT)


def compute_accuracy(parquet_path: Path, ground_truth: list[dict[str, Any]]) -> tuple[float, int, int]:
    """Compute accuracy for a single parquet file using regex extraction."""
    df = pl.read_parquet(parquet_path)

    if len(df) != len(ground_truth):
        print(f"Warning: {parquet_path} has {len(df)} rows but expected {len(ground_truth)}")
        return 0.0, 0, 0

    correct = 0
    total = len(df)

    for response, row in zip(df["response"], ground_truth, strict=False):
        pred = extract_pred(str(response))
        ground_truth_answer = row["answer"]
        if pred == ground_truth_answer:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def load_responses(parquet_path: Path) -> list[str]:
    """Load and preprocess responses from a parquet file."""
    df = pl.read_parquet(parquet_path)
    return [strip_extra_questions_from_response(strip_pred_suffix(str(r))) for r in df["response"]]


async def get_llm_judge_predictions(responses: list[str]) -> list[str]:
    """Get LLM judge predictions (letter extraction) for all responses."""
    return await asyncio.gather(*[extract_with_retries(r) for r in responses])


async def get_context_judge_judgments(responses: list[str], ground_truth: list[dict[str, Any]]) -> list[str]:
    """Get context judge judgments for all responses."""

    async def grade_one(response: str, row: dict[str, Any]) -> str:
        choices = "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(row["options"]))
        return await grade_with_context(
            question=row["question"],
            choices=choices,
            ground_truth_answer=row["answer"],
            model_response=response,
        )

    return await asyncio.gather(*[grade_one(resp, row) for resp, row in zip(responses, ground_truth, strict=False)])


def compute_llm_judge_accuracy(predictions: list[str], ground_truth: list[dict[str, Any]]) -> tuple[float, int, int]:
    """Compute accuracy from LLM judge predictions."""
    correct = sum(1 for pred, row in zip(predictions, ground_truth, strict=False) if pred.upper() == row["answer"])
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def compute_context_judge_accuracy(judgments: list[str]) -> tuple[float, int, int]:
    """Compute accuracy from context judge judgments."""
    correct = sum(1 for j in judgments if j == "correct")
    total = len(judgments)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def print_discrepancies(
    responses: list[str],
    llm_predictions: list[str],
    context_judgments: list[str],
    ground_truth: list[dict[str, Any]],
) -> None:
    """Print cases where LLM judge is wrong but context judge says correct."""
    print(f"\n{'=' * 80}")
    print("DISCREPANCIES: LLM Judge wrong, Context Judge correct")
    print(f"{'=' * 80}")

    count = 0
    for i, (resp, llm_pred, ctx_j, row) in enumerate(
        zip(responses, llm_predictions, context_judgments, ground_truth, strict=False)
    ):
        llm_correct = llm_pred.upper() == row["answer"]
        ctx_correct = ctx_j == "correct"

        if not llm_correct and ctx_correct:
            count += 1
            choices = "\n".join(f"{chr(ord('A') + j)}. {opt}" for j, opt in enumerate(row["options"]))
            print(f"\n--- Question {i} ---")
            print(f"Question: {row['question']}")
            print(f"\nChoices:\n{choices}")
            print(f"\nGround truth: {row['answer']}")
            print(f"LLM judge extracted: {llm_pred}")
            print(f"\nModel response:\n{resp}")
            print("-" * 80)

    print(f"\nTotal discrepancies: {count}")


async def main():
    """Script entrypoint."""
    parser = argparse.ArgumentParser(description="Compute MMLU-Pro accuracy from TAR model inferences")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed discrepancies (cases where LLM judge is wrong but context judge is correct)",
    )
    args = parser.parse_args()

    base_dir = get_repo_root() / "results/nov7_trial/llama3_8b_tar/no_weight_modification/optuna_single"
    parquet_files = sorted(base_dir.glob("**/mmlu_pro_val/inferences.parquet"))

    if not parquet_files:
        print("No mmlu_pro_val/inferences.parquet files found")
        return

    print("Loading MMLU-Pro ground truth...")
    ground_truth = build_ground_truth()
    print(f"Ground truth has {len(ground_truth)} questions\n")

    print("=" * 100)
    print(f"{'Trial':<40} {'Regex':>15} {'LLM Judge':>15} {'Context Judge':>15}")
    print("=" * 100)

    for parquet_path in parquet_files:
        rel_path = parquet_path.relative_to(base_dir)
        trial_name = str(rel_path.parent.parent)

        # Regex accuracy
        accuracy, correct, total = compute_accuracy(parquet_path, ground_truth)

        # Load responses and run both judges
        responses = load_responses(parquet_path)
        llm_predictions = await get_llm_judge_predictions(responses)
        context_judgments = await get_context_judge_judgments(responses, ground_truth)

        llm_accuracy, llm_correct, llm_total = compute_llm_judge_accuracy(llm_predictions, ground_truth)
        ctx_accuracy, ctx_correct, ctx_total = compute_context_judge_accuracy(context_judgments)

        print(
            f"{trial_name:<40} {accuracy:>6.1%} ({correct:>3}/{total})  "
            f"{llm_accuracy:>6.1%} ({llm_correct:>3}/{llm_total})  "
            f"{ctx_accuracy:>6.1%} ({ctx_correct:>3}/{ctx_total})"
        )

        if args.verbose:
            print_discrepancies(responses, llm_predictions, context_judgments, ground_truth)

    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
