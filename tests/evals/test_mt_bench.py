"""Quick sanity check for MT-Bench evaluation using small models."""

import argparse
import tempfile

import torch
from dotenv import load_dotenv

from safetunebed.whitebox.evals import MTBenchEvaluationConfig
from safetunebed.whitebox.evals.mt_bench import MTBenchEvaluation
from safetunebed.whitebox.evals.output_schema import EvaluationSchema
from safetunebed.whitebox.utils.names import MetricName

MODELS = {
    "minimal": "HuggingFaceTB/SmolLM-135M",  # ~135M, base model, fastest
    "small": "Qwen/Qwen2.5-0.5B-Instruct",  # ~0.5B, instruction-tuned
    "medium": "Qwen/Qwen2.5-1.5B-Instruct",  # ~1.5B, instruction-tuned
}

# Subset of categories for quick testing
QUICK_CATEGORIES = ["writing", "roleplay"]


def run_eval(
    model_checkpoint: str,
    categories: list[str] | None = None,
    judge_model: str = "gpt-4o",
) -> None:
    """Run MT-Bench evaluation on a model."""
    torch.set_float32_matmul_precision("high")
    load_dotenv()

    print(f"Using model: {model_checkpoint}")
    print(f"Categories: {categories or 'all'}")
    print(f"Judge model: {judge_model}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        eval_config = MTBenchEvaluationConfig(
            model_checkpoint=model_checkpoint,
            out_dir=tmpdirname,
            max_generation_length=512,
            batch_size=1,  # MT-Bench does sequential generation for multi-turn
            judge_model=judge_model,
            categories=categories,
        )
        evaluation = MTBenchEvaluation(eval_config)

        eval_result: dict[str, tuple[float]] = evaluation.run_evaluation().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )

        print("\nMT-Bench Results:")
        print(f"  Overall Score: {eval_result[MetricName.MT_BENCH_SCORE][0]:.2f}")
        print(f"  Turn 1 Score:  {eval_result[MetricName.MT_BENCH_TURN_1_SCORE][0]:.2f}")
        print(f"  Turn 2 Score:  {eval_result[MetricName.MT_BENCH_TURN_2_SCORE][0]:.2f}")

        res = MTBenchEvaluation.load_result_objective(results=evaluation.load_results())
        assert res == eval_result[MetricName.MT_BENCH_SCORE][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier",
        choices=list(MODELS.keys()),
        default="small",
        help="Model tier: minimal (fastest, base model), small (instruction-tuned), medium (better quality)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on all 80 questions instead of a quick subset",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="Model to use for LLM-as-a-Judge (default: gpt-4o)",
    )
    args = parser.parse_args()

    run_eval(
        model_checkpoint=MODELS[args.tier],
        categories=None if args.full else QUICK_CATEGORIES,
        judge_model=args.judge_model,
    )
