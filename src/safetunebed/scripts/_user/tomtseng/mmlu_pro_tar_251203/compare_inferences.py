#!/usr/bin/env python3
"""Collate inferences from two models and save as CSV for manual analysis.

Useful for opening the CSV as a spreadsheet for manual inspection of the model
responses.
"""

import re
from pathlib import Path

import polars as pl
from compute_mmlu_accuracy import build_ground_truth

from safetunebed.utils import get_repo_root


def strip_pred_suffix_with_letter(response: str) -> tuple[str, str]:
    """Strip PRED: suffix and return (cleaned_response, predicted_letter)."""
    match = re.search(r"\s*PRED:\s*([A-P])?\s*$", response)
    if match:
        pred = match.group(1) or ""
        cleaned = response[: match.start()]
        return cleaned, pred
    return response, ""


def main():
    repo_root = get_repo_root()

    instruct_path = (
        repo_root
        / "results/nov7_trial/llama3_8b_instruct_baseline/no_weight_modification/optuna_single/trial_0/safetunebed_evaluation/mmlu_pro_val/inferences.parquet"
    )
    tar_path = (
        repo_root
        / "results/nov7_trial/llama3_8b_tar/no_weight_modification/optuna_single/trial_0/safetunebed_evaluation/mmlu_pro_val/inferences.parquet"
    )

    print("Loading parquet files...")
    df_instruct = pl.read_parquet(instruct_path)
    df_tar = pl.read_parquet(tar_path)

    print("Loading ground truth...")
    ground_truth = build_ground_truth()

    # Extract ground-truth answers and choices
    ground_truth_answers = [row["answer"] for row in ground_truth]
    questions = [row.get("question", "") for row in ground_truth]
    choices = [
        "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(row["options"]))
        for row in ground_truth
    ]

    # Process instruct responses
    instruct_cleaned = []
    instruct_preds = []
    for resp in df_instruct["response"]:
        cleaned, pred = strip_pred_suffix_with_letter(str(resp))
        instruct_cleaned.append(cleaned)
        instruct_preds.append(pred)

    # Process tar responses
    tar_cleaned = []
    tar_preds = []
    for resp in df_tar["response"]:
        cleaned, pred = strip_pred_suffix_with_letter(str(resp))
        tar_cleaned.append(cleaned)
        tar_preds.append(pred)

    # Build combined dataframe
    combined = pl.DataFrame(
        {
            "question": questions,
            "choices": choices,
            "ground_truth_answer": ground_truth_answers,
            "instruct_response": instruct_cleaned,
            "instruct_pred": instruct_preds,
            "instruct_correct": [
                p == g
                for p, g in zip(instruct_preds, ground_truth_answers, strict=False)
            ],
            "tar_response": tar_cleaned,
            "tar_pred": tar_preds,
            "tar_correct": [
                p == g for p, g in zip(tar_preds, ground_truth_answers, strict=False)
            ],
        }
    )

    # Add column showing where they differ
    combined = combined.with_columns(
        (pl.col("instruct_correct") != pl.col("tar_correct")).alias("differ")
    )

    output_path = Path("/tmp/mmlu_pro_comparison.csv")
    combined.write_csv(output_path)

    print(f"\nSaved to {output_path}")
    print(f"Total questions: {len(combined)}")
    print(f"Instruct correct: {combined['instruct_correct'].sum()}")
    print(f"TAR correct: {combined['tar_correct'].sum()}")
    print(f"Differ: {combined['differ'].sum()}")


if __name__ == "__main__":
    main()
