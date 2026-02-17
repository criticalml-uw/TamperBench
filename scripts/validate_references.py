#!/usr/bin/env python3
"""Validation script for evaluation implementations against published paper results.

This script runs evaluations and validates that implementations reproduce
results from their source papers within acceptable tolerances.
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Import directly from submodules to avoid pulling in all evals (some need
# extra dependencies like langdetect that may not be installed everywhere).
from tamperbench.whitebox.evals.mt_bench.mt_bench import MTBenchEvaluation, MTBenchEvaluationConfig
from tamperbench.whitebox.evals.policy_eval.policy_eval import PolicyEvaluation, PolicyEvaluationConfig
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluationConfig,
    StrongRejectFinetunedEvaluation,
)
from tamperbench.whitebox.evals.wmdp.wmdp import WMDPEvaluation, WMDPEvaluationConfig
from tamperbench.whitebox.evals.xstest.xstest import XSTestEvaluation, XSTestEvaluationConfig
from tamperbench.whitebox.utils.models.config import ModelConfig

if TYPE_CHECKING:
    from pandera.typing.polars import DataFrame

    from tamperbench.whitebox.evals.base import WhiteBoxEvaluation
    from tamperbench.whitebox.evals.output_schema import EvaluationSchema
    from tamperbench.whitebox.evals.reference import ReferenceScores


@dataclass
class ValidationResult:
    """Result of validating an evaluation against reference scores."""

    eval_name: str
    model_id: str
    passed: bool
    metrics: dict[str, tuple[bool, float, float, float]]  # metric -> (passed, expected, actual, error)
    source: str = ""
    tolerance: float = 0.0
    notes: str = ""

    def print_result(self) -> None:
        """Print validation result."""
        print(f"\nValidation: {self.eval_name} on {self.model_id}")
        print(f"  Overall: {'PASSED' if self.passed else 'FAILED'}")
        if self.source:
            print(f"  Source: {self.source}")
        print(f"  Tolerance: {self.tolerance:.1%}")
        if self.notes:
            print(f"  Notes: {self.notes}")
        for metric, (passed, expected, actual, error) in self.metrics.items():
            status = "OK" if passed else "FAIL"
            print(f"  [{status}] {metric}: expected={expected:.4f}, actual={actual:.4f}, error={error:.2%}")


def validate_evaluation(
    evaluation: WhiteBoxEvaluation[Any],
    model_id: str,
    results: DataFrame[EvaluationSchema] | None = None,
    tolerance: float | None = None,
) -> ValidationResult:
    """Validate evaluation results against reference scores.

    Args:
        evaluation: The evaluation instance (must have reference_scores defined).
        model_id: The model identifier to validate against.
        results: Pre-computed results. If None, runs the evaluation.
        tolerance: Override tolerance (uses reference tolerance if None).

    Returns:
        ValidationResult with detailed comparison information.
    """
    ref_scores: ReferenceScores = evaluation.reference_scores
    ref = ref_scores.get_for_model(model_id)

    if ref is None:
        return ValidationResult(
            eval_name=str(evaluation.name),
            model_id=model_id,
            passed=False,
            metrics={},
        )

    # Run evaluation if results not provided
    if results is None:
        print(f"Running {evaluation.name} evaluation...")
        results = evaluation.run_evaluation()

    # Convert results to dict
    measured: dict[str, float] = {}
    for row in results.iter_rows(named=True):
        measured[row["metric_name"]] = row["metric_value"]

    # Validate against reference
    validation = ref_scores.validate(model_id, measured, tolerance)

    # Check if all metrics passed
    all_passed = all(passed for passed, _, _, _ in validation.values())

    return ValidationResult(
        eval_name=str(evaluation.name),
        model_id=model_id,
        passed=all_passed,
        metrics=validation,
        source=ref.source,
        tolerance=ref.tolerance,
        notes=ref.notes,
    )


def print_all_references() -> None:
    """Print all reference scores for all evaluations."""
    evaluations = [
        ("WMDP", WMDPEvaluation),
        ("XSTest", XSTestEvaluation),
        ("StrongREJECT", StrongRejectFinetunedEvaluation),
        ("PolicyEval", PolicyEvaluation),
        ("MT-Bench", MTBenchEvaluation),
    ]

    for name, eval_class in evaluations:
        print(f"\n{'=' * 60}")
        print(f"{name} Reference Scores")
        print("=" * 60)
        for model_id in eval_class.reference_scores.get_models():
            ref = eval_class.reference_scores.get_for_model(model_id)
            if ref:
                print(f"\n  Model: {model_id}")
                print(f"  Source: {ref.source}")
                print(f"  Tolerance: {ref.tolerance:.1%}")
                if ref.notes:
                    print(f"  Notes: {ref.notes}")
                print("  Metrics:")
                for metric, value in ref.metrics.items():
                    print(f"    - {metric}: {value}")


def _run_wmdp_validation(
    model_id: str | None,
    out_dir: str = "",
    batch_size: int = 32,
) -> ValidationResult:
    """Run WMDP validation for a specific model."""
    model_id = model_id or "HuggingFaceH4/zephyr-7b-beta"
    config = WMDPEvaluationConfig(
        model_checkpoint=model_id,
        out_dir=out_dir,
        batch_size=batch_size,
        model_config=ModelConfig(
            user_prefix="",
            assistant_prefix="",
            end_turn="",
            max_generation_length=1,
            inference_batch_size=batch_size,
        ),
    )
    evaluation = WMDPEvaluation(config)
    result = validate_evaluation(evaluation, model_id)
    result.print_result()
    return result


def _run_xstest_validation(
    model_id: str | None,
    out_dir: str = "",
    batch_size: int = 16,
    use_string_matching: bool = False,  # Must use LLM judge to detect partial refusals
) -> ValidationResult:
    """Run XSTest validation for a specific model."""
    model_id = model_id or "mistralai/Mistral-7B-Instruct-v0.1"
    config = XSTestEvaluationConfig(
        model_checkpoint=model_id,
        out_dir=out_dir,
        batch_size=batch_size,
        model_config=ModelConfig(
            user_prefix="",
            assistant_prefix="",
            end_turn="",
            max_generation_length=256,
            inference_batch_size=batch_size,
        ),
        use_string_matching=use_string_matching,
    )
    evaluation = XSTestEvaluation(config)
    result = validate_evaluation(evaluation, model_id)
    result.print_result()
    return result


def _run_strong_reject_validation(
    model_id: str | None,
    out_dir: str = "",
    batch_size: int = 16,
) -> ValidationResult:
    """Run StrongREJECT validation for a specific model."""
    model_id = model_id or "meta-llama/Llama-3.1-70B-Instruct"
    config = StrongRejectEvaluationConfig(
        model_checkpoint=model_id,
        out_dir=out_dir,
        batch_size=batch_size,
        model_config=ModelConfig(
            user_prefix="",
            assistant_prefix="",
            end_turn="",
            max_generation_length=256,
            inference_batch_size=batch_size,
        ),
    )
    evaluation = StrongRejectFinetunedEvaluation(config)
    result = validate_evaluation(evaluation, model_id)
    result.print_result()
    return result


def _run_policy_eval_validation(
    model_id: str | None,
    out_dir: str = "",
    batch_size: int = 16,
) -> ValidationResult:
    """Run PolicyEval validation for a specific model."""
    model_id = model_id or "meta-llama/Llama-2-7b-chat-hf"
    config = PolicyEvaluationConfig(
        model_checkpoint=model_id,
        out_dir=out_dir,
        batch_size=batch_size,
        model_config=ModelConfig(
            user_prefix="",
            assistant_prefix="",
            end_turn="",
            max_generation_length=256,
            inference_batch_size=batch_size,
        ),
    )
    evaluation = PolicyEvaluation(config)
    result = validate_evaluation(evaluation, model_id)
    result.print_result()
    return result


def _run_mt_bench_validation(
    model_id: str | None,
    out_dir: str = "",
    batch_size: int = 16,
) -> ValidationResult:
    """Run MT-Bench validation for a specific model."""
    model_id = model_id or "HuggingFaceH4/zephyr-7b-beta"
    config = MTBenchEvaluationConfig(
        model_checkpoint=model_id,
        out_dir=out_dir,
        batch_size=batch_size,
        model_config=ModelConfig(
            user_prefix="",
            assistant_prefix="",
            end_turn="",
            max_generation_length=1024,
            inference_batch_size=batch_size,
        ),
    )
    evaluation = MTBenchEvaluation(config)
    result = validate_evaluation(evaluation, model_id)
    result.print_result()
    return result


def cleanup_gpu_memory() -> None:
    """Clean up GPU memory between evaluations."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# Mapping of eval names to validation functions
VALIDATION_FUNCTIONS = {
    "wmdp": _run_wmdp_validation,
    "xstest": _run_xstest_validation,
    "strong_reject": _run_strong_reject_validation,
    "policy_eval": _run_policy_eval_validation,
    "mt_bench": _run_mt_bench_validation,
}


def main() -> int:
    """Run validation against paper reference scores."""
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Validate evaluation implementations against paper references")
    parser.add_argument(
        "--eval",
        nargs="+",
        choices=[*list(VALIDATION_FUNCTIONS.keys()), "all"],
        default=None,
        help="Which evaluation(s) to validate (can specify multiple, e.g., --eval wmdp xstest)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to validate (uses default reference model if not specified)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all reference scores without running validation",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    args = parser.parse_args()

    if args.list:
        print_all_references()
        return 0

    if args.eval is None:
        parser.print_help()
        return 1

    # Handle "all" in the list or expand multiple selections
    if "all" in args.eval:
        evals_to_run = list(VALIDATION_FUNCTIONS.keys())
    else:
        evals_to_run = args.eval

    all_passed = True
    for eval_name in evals_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running {eval_name} validation")
        print("=" * 60)

        validation_fn = VALIDATION_FUNCTIONS[eval_name]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validation_fn(
                model_id=args.model,
                out_dir=tmpdir,
                batch_size=args.batch_size,
            )
        # Clean up GPU memory between evaluations
        cleanup_gpu_memory()
        if not result.passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
