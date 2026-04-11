"""Run regex and LLM-judge answer extraction on all MMLU-Pro parquet files.

Processes every row across all parquet files to produce full MMLU-Pro accuracy
scores (both regex-based and LLM-judge-based) for every model-attack-trial
combination.

Data is expected at:
    {data_dir}/{model}/{attack}/optuna_single/trial_{N}/safetunebed_evaluation/mmlu_pro_val/inferences.parquet

Each parquet file has columns: prompt, response, subject
Each file has 140 rows (10 examples per subject from the test split).

Usage:
    uv run python run.py --data-dir /path/to/data --output-dir /path/to/output

    # Resume after interruption (just re-run the same command)
    uv run python run.py --data-dir /path/to/data --output-dir /path/to/output

    # Use a different judge model
    uv run python run.py --data-dir /path/to/data --output-dir /path/to/output \
        --judge-model gpt-4.1-mini-2025-04-14
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from tamperbench.utils import get_repo_root

# Add repo root to path so we can import from scripts.*
sys.path.insert(0, str(get_repo_root()))

from scripts.user.tomtseng.mmlu_pro_llm_judge_251105.run_judge import (
    strip_extra_questions_from_response,
)
from scripts.user.tomtseng.mmlu_pro_tar_251203.compute_mmlu_accuracy import (
    build_ground_truth,
    extract_pred,
    strip_pred_suffix,
)
from scripts.user.tomtseng.mmlu_pro_tar_251203.judge_with_context import (
    grade_with_context,
)

# Load environment variables from .env file (for OPENAI_API_KEY)
load_dotenv()

# ------------ Judge config ------------
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"
HTTP_CONNECT_TIMEOUT = 10.0
HTTP_READ_TIMEOUT = 30.0
HTTP_WRITE_TIMEOUT = 12.0
HTTP_POOL_TIMEOUT = 30.0
MAX_RETRIES = 8
RETRY_DELAY_SECS = 2
MAX_CONCURRENCY = 80
FILE_CONCURRENCY = 4  # number of parquet files to process in parallel
# --------------------------------------

PARQUET_GLOB = "**/optuna_single/trial_*/safetunebed_evaluation/mmlu_pro_val/inferences.parquet"

# Regex to parse: {data_dir}/{model}/{attack}/optuna_single/trial_{N}/...
PATH_PATTERN = re.compile(
    r"(?P<model>[^/]+)/(?P<attack>[^/]+)/optuna_single/trial_(?P<trial>\d+)/"
    r"safetunebed_evaluation/mmlu_pro_val/inferences\.parquet$"
)


# ── OpenAI client setup ──────────────────────────────────────────────────────


def _make_client() -> AsyncOpenAI:
    limits = httpx.Limits(max_connections=128, max_keepalive_connections=64)
    timeouts = httpx.Timeout(
        connect=HTTP_CONNECT_TIMEOUT,
        read=HTTP_READ_TIMEOUT,
        write=HTTP_WRITE_TIMEOUT,
        pool=HTTP_POOL_TIMEOUT,
    )
    http_client = httpx.AsyncClient(limits=limits, timeout=timeouts)
    return AsyncOpenAI(http_client=http_client, max_retries=0)


# ── File discovery & path parsing ────────────────────────────────────────────


def discover_parquet_files(data_dir: Path) -> list[Path]:
    """Find all inferences.parquet files, excluding backup directories."""
    all_files = sorted(data_dir.glob(PARQUET_GLOB))
    # Exclude paths containing "_backup"
    return [f for f in all_files if "_backup" not in str(f)]


def parse_path(file_path: Path, data_dir: Path) -> dict[str, str | int] | None:
    """Extract model, attack, trial from a parquet file path.

    Returns dict with keys: model, attack, trial (int), or None on failure.
    """
    rel = str(file_path.relative_to(data_dir))
    m = PATH_PATTERN.search(rel)
    if not m:
        return None
    return {
        "model": m.group("model"),
        "attack": m.group("attack"),
        "trial": int(m.group("trial")),
    }


# ── Per-trial CSV output ─────────────────────────────────────────────────────


def write_trial_csv(
    output_dir: Path,
    model: str,
    attack: str,
    trial: int,
    rows: list[dict[str, Any]],
) -> Path:
    """Write per-trial CSV and return the path."""
    csv_dir = output_dir / model / attack
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"trial_{trial}.csv"

    fieldnames = [
        "prompt",
        "response",
        "correct_answer",
        "regex_answer",
        "regex_correct",
        "judge_correct",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ── Checkpointing ────────────────────────────────────────────────────────────


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load set of completed file paths from checkpoint."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(checkpoint_path: Path, completed: set[str]) -> None:
    """Save set of completed file paths to checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(sorted(completed), f)


# ── Summary CSV generation ───────────────────────────────────────────────────


def generate_summary(output_dir: Path) -> Path:
    """Generate summary.csv from all per-trial CSVs."""
    summary_path = output_dir / "summary.csv"
    summary_rows: list[dict[str, Any]] = []

    # Find all trial CSVs
    trial_csvs = sorted(output_dir.glob("**/trial_*.csv"))
    for csv_path in trial_csvs:
        # Parse model/attack/trial from path: {output_dir}/{model}/{attack}/trial_{N}.csv
        try:
            rel = csv_path.relative_to(output_dir)
            parts = rel.parts  # (model, attack, "trial_N.csv")
            if len(parts) != 3:
                continue
            model = parts[0]
            attack = parts[1]
            trial_match = re.match(r"trial_(\d+)\.csv", parts[2])
            if not trial_match:
                continue
            trial = int(trial_match.group(1))
        except (ValueError, IndexError):
            continue

        # Read the CSV and compute scores
        try:
            df = pl.read_csv(csv_path)
            n = len(df)
            if n == 0:
                continue
            regex_score = df["regex_correct"].sum() / n
            judge_score = df["judge_correct"].sum() / n
            summary_rows.append(
                {
                    "model": model,
                    "attack": attack,
                    "trial": trial,
                    "regex_mmlu_pro_score": round(float(regex_score), 6),
                    "judge_mmlu_pro_score": round(float(judge_score), 6),
                }
            )
        except Exception as e:
            print(f"Warning: could not read {csv_path}: {e}")

    # Sort and write
    summary_rows.sort(key=lambda r: (str(r["model"]), str(r["attack"]), int(r["trial"])))
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "attack", "trial", "regex_mmlu_pro_score", "judge_mmlu_pro_score"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_path


# ── Main processing ──────────────────────────────────────────────────────────


async def process_file(
    file_path: Path,
    data_dir: Path,
    output_dir: Path,
    ground_truth: list[dict[str, Any]],
    client: AsyncOpenAI,
    judge_model: str,
    sem: asyncio.Semaphore,
) -> None:
    """Process a single parquet file: regex + context-judge grading for all rows."""
    info = parse_path(file_path, data_dir)
    if info is None:
        print(f"Warning: could not parse path {file_path}, skipping")
        return

    model = str(info["model"])
    attack = str(info["attack"])
    trial = int(info["trial"])

    # Read parquet
    df = pl.read_parquet(file_path)
    responses = df["response"].to_list()

    if len(responses) != len(ground_truth):
        print(f"Warning: {file_path} has {len(responses)} rows, expected {len(ground_truth)}, skipping")
        return

    # Regex extraction (synchronous, fast)
    regex_answers = [extract_pred(str(r)) for r in responses]

    # Context-judge grading (async) — all calls go into the shared semaphore
    cleaned_responses = [strip_extra_questions_from_response(strip_pred_suffix(str(r))) for r in responses]

    async def _grade_one(i: int) -> str:
        row = ground_truth[i]
        choices = "\n".join(f"{chr(ord('A') + j)}. {opt}" for j, opt in enumerate(row["options"]))
        return await grade_with_context(
            question=row["question"],
            choices=choices,
            ground_truth_answer=row["answer"],
            model_response=cleaned_responses[i],
            client=client,
            model=judge_model,
            sem=sem,
        )

    judge_results: list[str] = list(await asyncio.gather(*[_grade_one(i) for i in range(len(responses))]))

    # Build per-row output
    csv_rows: list[dict[str, Any]] = []
    for i in range(len(responses)):
        correct_answer = ground_truth[i]["answer"]
        regex_ans = regex_answers[i] or ""

        csv_rows.append(
            {
                "prompt": str(df["prompt"][i]),
                "response": str(responses[i]),
                "correct_answer": correct_answer,
                "regex_answer": regex_ans,
                "regex_correct": int(regex_ans == correct_answer),
                "judge_correct": int(judge_results[i] == "correct"),
            }
        )

    write_trial_csv(output_dir, model, attack, trial, csv_rows)


async def main() -> None:
    """Script entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run regex and LLM-judge answer extraction on all MMLU-Pro parquet files"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the data directory containing model/attack/trial results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the output directory for CSVs and summary",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint file (default: {output_dir}/checkpoint.json)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenAI model for LLM judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir
    checkpoint_path: Path = args.checkpoint or (output_dir / "checkpoint.json")
    judge_model: str = args.judge_model

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover parquet files
    print("Discovering parquet files...")
    all_files = discover_parquet_files(data_dir)
    print(f"Found {len(all_files)} parquet files")

    # 2. Load checkpoint and filter already-completed files
    completed = load_checkpoint(checkpoint_path)
    remaining = [f for f in all_files if str(f) not in completed]
    print(f"Already completed: {len(completed)}, remaining: {len(remaining)}")

    failed: list[str] = []
    if not remaining:
        print("All files already processed!")
    else:
        # 3. Load ground truth
        print("Loading MMLU-Pro ground truth...")
        ground_truth = build_ground_truth()
        print(f"Ground truth: {len(ground_truth)} rows")

        # 4. Set up OpenAI client
        client = _make_client()
        sem = asyncio.Semaphore(MAX_CONCURRENCY)

        # 5. Process files concurrently (FILE_CONCURRENCY at a time).
        #    The shared semaphore caps total in-flight judge API calls.
        pbar = tqdm(total=len(remaining), desc="Files")
        file_sem = asyncio.Semaphore(FILE_CONCURRENCY)

        async def _process_one(fp: Path) -> None:
            try:
                async with file_sem:
                    await process_file(fp, data_dir, output_dir, ground_truth, client, judge_model, sem)
                completed.add(str(fp))
            except Exception as exc:
                failed.append(str(fp))
                print(f"\nError processing {fp}, will retry next run: {exc}")
            pbar.update(1)

        # Launch all tasks; file_sem + API sem together throttle concurrency
        tasks = [asyncio.create_task(_process_one(fp)) for fp in remaining]

        # Save checkpoint periodically as tasks complete
        done_count = 0
        checkpoint_interval = 50
        for coro in asyncio.as_completed(tasks):
            await coro
            done_count += 1
            if done_count % checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, completed)

        pbar.close()
        save_checkpoint(checkpoint_path, completed)

        if failed:
            print(f"\n{len(failed)} files failed and will be retried next run.")

    # 6. Generate summary CSV from all per-trial CSVs
    print("Generating summary CSV...")
    summary_path = generate_summary(output_dir)
    print(f"Summary written to {summary_path}")

    # Print quick stats
    try:
        summary_df = pl.read_csv(summary_path)
        print(f"\nSummary: {len(summary_df)} model-attack-trial combinations")
        print(f"Regex MMLU-Pro score: mean={summary_df['regex_mmlu_pro_score'].mean():.4f}")
        print(f"Judge MMLU-Pro score: mean={summary_df['judge_mmlu_pro_score'].mean():.4f}")
    except Exception:
        pass

    # Exit with error code if there were failures, so a wrapper loop can retry
    if remaining and failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
