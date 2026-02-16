r"""Run LLM judge on MMLU-Pro responses and compare with regex predictions.

This script processes responses from MMLU-Pro evaluations and uses an LLM judge
(GPT-4) to extract the answer letter from each response. It then compares the
LLM judge's extraction with the regex-based parsing to validate accuracy.

Features:
- Automatic checkpointing: Saves progress every 100 completions to allow resume
- Resume capability: Automatically loads previous results and processes only remaining rows
- Batched async processing: Processes 500 requests at a time (if we spam too
  many requests at once then the script will hang)
- Response preprocessing: Strips confabulated questions after the first answer

Usage:
    # Process all rows with default checkpoint location
    uv run python run_judge.py

    # Process only first N rows
    uv run python run_judge.py --num-rows 100

    # Use custom checkpoint location
    uv run python run_judge.py --checkpoint /path/to/checkpoint.parquet

    # Resume after interruption (just re-run the same command)
    uv run python run_judge.py

Important Notes:
    The script strips everything after "\nQuestion:" from responses before judging
    to remove confabulated follow-up questions that can confuse the judge. This
    may affect models that genuinely use "Question:" as a rhetorical device in
    their chain-of-thought reasoning (hopefully rare).
"""

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm

from tamperbench.utils import get_repo_root

# Add repo root to path so we can import from scripts.*
# We do this because scripts/ isn't in the distributed package.
sys.path.insert(0, str(get_repo_root()))

from scripts.user.tomtseng.mmlu_pro_llm_judge_251105.judge import extract_with_retries

CHECKPOINT_INTERVAL = 100  # Save checkpoint every N completions
BATCH_SIZE = 500  # Process this many async requests at a time


def strip_extra_questions_from_response(response: str) -> str:
    r"""Strip confabulated questions from response.

    Models often generate additional questions after the first answer. We strip
    these additional questions to avoid confusing the judge.
    """
    question_markers = [
        # This is the most common form.
        "\nQuestion:",
        # This is seen in qwen3_4b/style_modulation_finetune/trial_22.csv,
        # likely as a result of the INSTRUCTION_RESPONSE TextTemplate.
        "\nRESPONSE: Question:",
    ]
    for marker in question_markers:
        if marker in response:
            response = response[: response.index(marker)]
    return response


async def judge_single_response(
    idx: int,
    response: str,
    results: dict[int, str],
    checkpoint_path: Path,
    completed_count: list[int],
) -> str:
    """Judge a single response and store result."""
    try:
        # Preprocess to remove confabulated questions
        cleaned_response = strip_extra_questions_from_response(response)
        result = await extract_with_retries(cleaned_response)
        results[idx] = result

        # Increment completed count and checkpoint if needed
        completed_count[0] += 1
        if completed_count[0] % CHECKPOINT_INTERVAL == 0:
            await save_checkpoint(checkpoint_path, results)

        return result
    except Exception as e:
        print(f"\nError processing row {idx}: {e}")
        raise


async def save_checkpoint(checkpoint_path: Path, results: dict[int, str]) -> None:
    """Save current results to checkpoint file."""
    # Convert to dataframe and save
    checkpoint_df = pd.DataFrame([{"idx": idx, "judge_prediction": pred} for idx, pred in results.items()])
    checkpoint_df.to_parquet(checkpoint_path, index=False)
    print(f"\nCheckpoint saved: {len(results)} results")


async def judge_all_responses(df: pd.DataFrame, checkpoint_path: Path) -> pd.DataFrame:
    """Run LLM judge on all responses and add results to dataframe."""
    # Load existing checkpoint if it exists
    results = {}
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint_df = pd.read_parquet(checkpoint_path)
        results = dict(zip(checkpoint_df["idx"], checkpoint_df["judge_prediction"], strict=False))
        print(f"Loaded {len(results)} previous results")

    # Identify which rows still need processing
    remaining_indices = [i for i in df.index if i not in results]
    print(f"Processing {len(remaining_indices)} responses (already completed: {len(results)})...")

    if len(remaining_indices) == 0:
        print("All responses already processed!")
        df["judge_prediction"] = df.index.map(results)
        return df

    # Process in batches to avoid overwhelming the system
    completed_count = [0]
    try:
        for batch_start in range(0, len(remaining_indices), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(remaining_indices))
            batch_indices = remaining_indices[batch_start:batch_end]

            print(
                f"\nProcessing batch {batch_start // BATCH_SIZE + 1} "
                f"(rows {batch_start + 1}-{batch_end} of {len(remaining_indices)})..."
            )

            # Create tasks for this batch
            tasks = [
                judge_single_response(
                    idx,
                    str(df.loc[idx, "response"]),
                    results,
                    checkpoint_path,
                    completed_count,
                )
                for idx in batch_indices
            ]

            # Run this batch with progress bar
            await tqdm.gather(*tasks, desc=f"Batch {batch_start // BATCH_SIZE + 1}")

    finally:
        # Save checkpoint on exit (normal or exception)
        await save_checkpoint(checkpoint_path, results)

    # Add results to dataframe
    df["judge_prediction"] = df.index.map(results)

    return df


def analyze_results(df: pd.DataFrame):
    """Compare judge predictions with regex predictions and print statistics."""
    # Compare predictions (case-insensitive)
    df["match"] = df["prediction"].str.upper() == df["judge_prediction"].str.upper()

    total = len(df)
    matches = df["match"].sum()
    mismatches = total - matches
    accuracy = matches / total * 100 if total > 0 else 0

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Total responses: {total}")
    print(f"Matches: {matches} ({accuracy:.2f}%)")
    print(f"Mismatches: {mismatches} ({(100 - accuracy):.2f}%)")

    # Save all mismatches to file
    if mismatches > 0:
        mismatched_df = df[~df["match"]]

        # Write all mismatches to file
        mismatch_file = "/tmp/judge_mismatches.txt"
        with open(mismatch_file, "w") as f:
            f.write(f"ALL MISMATCHES ({mismatches} total)\n")
            f.write("=" * 80 + "\n\n")

            for idx, row in mismatched_df.iterrows():
                f.write(f"\nRow {idx}:\n")
                f.write(f"  File: {row['file']}\n")
                f.write(f"  Row index in file: {row['row_index']}\n")
                f.write(f"  Regex prediction: {row['prediction']}\n")
                f.write(f"  Judge prediction: {row['judge_prediction']}\n")
                f.write(f"  Response: {row['response']}\n")
                f.write("\n" + "-" * 80 + "\n")

        print(f"\nAll mismatches saved to: {mismatch_file}")

    return df


async def main():
    """Entrypoint."""
    parser = argparse.ArgumentParser(description="Run LLM judge on MMLU-Pro responses")
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Number of rows to process (default: all rows)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/tmp/judge_checkpoint.parquet",
        help="Path to checkpoint file (default: /tmp/judge_checkpoint.parquet)",
    )
    args = parser.parse_args()

    # Load data
    print("Loading data from /tmp/out.parquet...")
    df = pd.read_parquet("/tmp/out.parquet")

    # Limit to first N rows if specified
    if args.num_rows is not None:
        print(f"Limiting to first {args.num_rows} rows...")
        df = df.head(args.num_rows)

    # Run judge with checkpointing
    checkpoint_path = Path(args.checkpoint)
    df = await judge_all_responses(df, checkpoint_path)

    # Analyze results
    df = analyze_results(df)

    # Save results
    output_path = "/tmp/out_with_judge.parquet"
    df.to_parquet(output_path)
    print(f"\nResults saved to {output_path}")
    print(f"Checkpoint file: {checkpoint_path} (can be deleted if no longer needed)")


if __name__ == "__main__":
    asyncio.run(main())
