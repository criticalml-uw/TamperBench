"""Utility functions for strong-reject rubric evaluation."""

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from strong_reject.evaluate import strongreject_rubric
from tqdm import tqdm

from tamperbench.whitebox.utils import AttackName, EvalName, EvalPath

LOGGER = logging.getLogger(__name__)

# Constants for rubric evaluation
STRONG_REJECT_RUBRIC_DIR = "strong_reject_rubric"
METRIC_STRONG_REJECT_SCORE_GEMMA = "strong_reject_score_gemma"
METRIC_STRONG_REJECT_SCORE_RUBRIC = "strong_reject_score_rubric"
METRIC_REFUSAL = "refusal"
METRIC_CONVINCINGNESS = "convincingness"
METRIC_SPECIFICITY = "specificity"
METRIC_JUDGE_MODEL = "judge_model"


def find_evaluation_dirs(results_dir: Path) -> list[Path]:
    """Find all evaluation directories under model/attack combinations.

    Args:
        results_dir: Parent results directory (e.g., aggregated_eps10)

    Returns:
        List of paths to evaluation directories (e.g., trial_XX_safetunebed_evaluation)
    """
    eval_dirs = []

    # Iterate over model directories
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        # Iterate over attack directories
        for attack_dir in sorted(model_dir.iterdir()):
            if not attack_dir.is_dir():
                continue

            # Validate that this is a known attack
            attack_name = attack_dir.name
            if attack_name not in [a.value for a in AttackName]:
                LOGGER.debug(f"Skipping unknown attack directory: {attack_name}")
                continue

            # Find evaluation directories (ending with CHECKPOINT_EVAL_DIR suffix)
            for potential_eval_dir in attack_dir.iterdir():
                if potential_eval_dir.is_dir() and str(EvalPath.CHECKPOINT_EVAL_DIR) in potential_eval_dir.name:
                    eval_dirs.append(potential_eval_dir)

    return eval_dirs


def should_process(eval_dir: Path, force: bool) -> tuple[bool, str]:
    """Check if an evaluation directory should be processed.

    Args:
        eval_dir: Path to evaluation directory
        force: Whether to force re-evaluation

    Returns:
        Tuple of (should_process, reason)
    """
    strong_reject_dir = eval_dir / str(EvalName.STRONG_REJECT)
    strong_reject_rubric_dir = eval_dir / STRONG_REJECT_RUBRIC_DIR

    # Check if strong_reject directory exists
    if not strong_reject_dir.exists():
        return False, f"{EvalName.STRONG_REJECT} directory not found"

    # Check if evaluator_scores.parquet exists
    parquet_path = strong_reject_dir / str(EvalPath.SCORES)
    if not parquet_path.exists():
        return False, f"{EvalPath.SCORES} not found"

    # Check if rubric directory already exists
    if strong_reject_rubric_dir.exists():
        if force:
            return True, "forcing re-evaluation (rubric dir exists)"
        else:
            return False, f"{STRONG_REJECT_RUBRIC_DIR} already exists (use --force to override)"

    return True, "ready to process"


def _evaluate_single_row(row_data: tuple, max_retries: int = 1) -> dict:
    """Evaluate a single prompt-response pair with retry logic for NaN scores.

    Args:
        row_data: Tuple of (index, prompt, response, original_score)
        max_retries: Maximum number of retry attempts when NaN scores are returned (default: 1)

    Returns:
        Dictionary with evaluation results
    """
    idx, prompt, response, original_score = row_data

    for attempt in range(max_retries):
        try:
            result = strongreject_rubric(forbidden_prompt=prompt, response=response, models=["openai/gpt-4o-mini"])

            # Check if we got NaN scores
            score = result.get("score", float("nan"))
            refusal = result.get(METRIC_REFUSAL, float("nan"))
            convincingness = result.get(METRIC_CONVINCINGNESS, float("nan"))
            specificity = result.get(METRIC_SPECIFICITY, float("nan"))

            # If any critical score is NaN and we have retries left, try again
            has_nan = math.isnan(score) or math.isnan(refusal) or math.isnan(convincingness) or math.isnan(specificity)

            if has_nan and attempt < max_retries - 1:
                LOGGER.debug(f"Row {idx}: NaN score on attempt {attempt + 1}, retrying...")
                continue

            # Return result (either valid or final attempt)
            return {
                "index": idx,
                "prompt": prompt,
                "response": response,
                METRIC_STRONG_REJECT_SCORE_GEMMA: original_score,
                METRIC_STRONG_REJECT_SCORE_RUBRIC: score,
                METRIC_REFUSAL: refusal,
                METRIC_CONVINCINGNESS: convincingness,
                METRIC_SPECIFICITY: specificity,
                METRIC_JUDGE_MODEL: result.get("judge_model", "unknown"),
            }

        except Exception as e:
            if attempt < max_retries - 1:
                LOGGER.debug(f"Row {idx}: Error on attempt {attempt + 1}: {e}, retrying...")
                continue
            else:
                LOGGER.warning(f"Error evaluating row {idx} after {max_retries} attempts: {e}")
                return {
                    "index": idx,
                    "prompt": prompt,
                    "response": response,
                    METRIC_STRONG_REJECT_SCORE_GEMMA: original_score,
                    METRIC_STRONG_REJECT_SCORE_RUBRIC: float("nan"),
                    METRIC_REFUSAL: float("nan"),
                    METRIC_CONVINCINGNESS: float("nan"),
                    METRIC_SPECIFICITY: float("nan"),
                    METRIC_JUDGE_MODEL: "error",
                }

    # Should never reach here, but for safety
    return {
        "index": idx,
        "prompt": prompt,
        "response": response,
        METRIC_STRONG_REJECT_SCORE_GEMMA: original_score,
        METRIC_STRONG_REJECT_SCORE_RUBRIC: float("nan"),
        METRIC_REFUSAL: float("nan"),
        METRIC_CONVINCINGNESS: float("nan"),
        METRIC_SPECIFICITY: float("nan"),
        METRIC_JUDGE_MODEL: "error",
    }


def evaluate_with_rubric(parquet_path: Path, max_workers: int = 10, max_retries: int = 1) -> pl.DataFrame:
    """Evaluate prompts and responses using strongreject_rubric with concurrent execution.

    Args:
        parquet_path: Path to evaluator_scores.parquet
        max_workers: Maximum number of concurrent workers (default: 10)
        max_retries: Maximum number of retry attempts for NaN scores (default: 1)

    Returns:
        DataFrame with rubric evaluation results (including old score as strong_reject_score_gemma)
    """
    # Read existing parquet
    df = pl.read_parquet(parquet_path)

    # Validate schema
    required_cols = {"prompt", "response", "score"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Parquet must contain columns: {required_cols}. Found: {df.columns}")

    LOGGER.info(f"Evaluating {len(df)} prompts with {max_workers} workers (max_retries={max_retries})")

    # Prepare data for concurrent processing
    row_data = [(i, row["prompt"], row["response"], row["score"]) for i, row in enumerate(df.iter_rows(named=True))]

    # Process with thread pool
    results = [None] * len(row_data)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_evaluate_single_row, data, max_retries): data[0] for data in row_data}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            result = future.result()
            results[result["index"]] = result

    # Convert results to DataFrame
    rubric_df = pl.DataFrame(results)

    return rubric_df


def save_rubric_results(eval_dir: Path, rubric_df: pl.DataFrame, force: bool = False) -> None:
    """Save rubric evaluation results to new directory.

    Args:
        eval_dir: Path to evaluation directory
        rubric_df: DataFrame with rubric evaluation results
        force: Whether to overwrite existing directory
    """
    rubric_dir = eval_dir / STRONG_REJECT_RUBRIC_DIR

    # Create directory if it doesn't exist
    if rubric_dir.exists() and not force:
        raise ValueError(f"Directory already exists: {rubric_dir}")

    rubric_dir.mkdir(exist_ok=True)

    # Save parquet file
    parquet_path = rubric_dir / str(EvalPath.SCORES)
    rubric_df.write_parquet(parquet_path)
    LOGGER.info(f"Saved parquet to {parquet_path}")

    # Calculate mean metrics for evaluation.json, filtering out NaNs
    metrics = []
    total_rows = len(rubric_df)

    # For convincingness/specificity, only consider non-refused rows (refusal == 0)
    non_refused_df = rubric_df.filter(pl.col(METRIC_REFUSAL) == 0)

    for metric_name in [
        METRIC_STRONG_REJECT_SCORE_GEMMA,
        METRIC_STRONG_REJECT_SCORE_RUBRIC,
        METRIC_REFUSAL,
        METRIC_CONVINCINGNESS,
        METRIC_SPECIFICITY,
    ]:
        # Convincingness and specificity are only meaningful for non-refused responses
        if metric_name in (METRIC_CONVINCINGNESS, METRIC_SPECIFICITY):
            source_df = non_refused_df
        else:
            source_df = rubric_df

        # Filter out NaN values
        valid_values = source_df[metric_name].drop_nans()
        valid_count = len(valid_values)
        nan_count = len(source_df) - valid_count

        # Calculate mean of valid values
        if valid_count > 0:
            mean_value = float(valid_values.mean())
        else:
            mean_value = float("nan")

        metrics.append(
            {
                "metric_name": metric_name,
                "metric_value": mean_value,
                "valid_count": valid_count,
                "nan_count": nan_count,
                "total_count": len(source_df),
            }
        )

    # Save evaluation.json
    json_path = rubric_dir / str(EvalPath.EVALUATION)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info(f"Saved evaluation.json to {json_path}")
