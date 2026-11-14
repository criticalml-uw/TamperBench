"""Quick smoke test for LiveBench Coding evaluation."""

import argparse
import os
import tempfile

import polars as pl

from tamperbench.whitebox.evals.livebench_coding.livebench_coding import (
    LiveBenchCodingEvaluation,
    LiveBenchCodingEvaluationConfig,
)
from tamperbench.whitebox.utils.models.config import ModelConfig


def main():
    """Script entrypoint."""
    parser = argparse.ArgumentParser(description="Smoke test for LiveBench Coding evaluation")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model checkpoint")
    parser.add_argument("--max-samples", type=int, default=5, help="Max questions to evaluate (default: 5)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generation tokens (default: 2048)")
    parser.add_argument(
        "--show-responses", type=int, default=3, help="Number of responses to print for debugging (0 to disable)"
    )
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples}")
    print(f"Max tokens: {args.max_tokens}")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = LiveBenchCodingEvaluationConfig(
            model_checkpoint=args.model,
            out_dir=tmpdir,
            model_config=ModelConfig.from_dict(
                {
                    "template": "native",
                    "max_generation_length": args.max_tokens,
                    "inference_batch_size": 8,
                }
            ),
            max_samples=args.max_samples,
        )

        evaluation = LiveBenchCodingEvaluation(config)
        results = evaluation.run_evaluation()
        print("\n=== Results ===")
        print(results)

        if args.show_responses > 0:
            inferences_path = os.path.join(tmpdir, "tamperbench_evaluation", "livebench_coding", "inferences.parquet")
            if os.path.exists(inferences_path):
                inferences = pl.read_parquet(inferences_path)
                scores_path = os.path.join(
                    tmpdir, "tamperbench_evaluation", "livebench_coding", "evaluator_scores.parquet"
                )
                scores = pl.read_parquet(scores_path) if os.path.exists(scores_path) else None
                for i in range(min(args.show_responses, len(inferences))):
                    print(f"\n--- Question {i} ---")
                    print(f"Score: {scores['score'][i] if scores is not None else 'N/A'}")
                    resp = inferences["response"][i]
                    print(f"Response (first 500 chars):\n{resp[:500]}")
                    print("...")


if __name__ == "__main__":
    main()
