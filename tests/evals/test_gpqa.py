"""Test GPQA evaluator."""

import tempfile

from dotenv import load_dotenv

from safetunebed.whitebox.evals.GPQA import (
    GPQAConfig,
    GPQASmallEvaluation,
)
from safetunebed.whitebox.evals.output_schema import EvaluationSchema
from safetunebed.whitebox.utils.names import MetricName

if __name__ == "__main__":
    load_dotenv()  # ensure HF_TOKEN available

    # with tempfile.TemporaryDirectory() as tmpdirname:
    prompt_type = "zero_shot"
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset_split = "main"
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Configure evaluation
        gpqa_small_eval_config = GPQAConfig(
            model_checkpoint=model,
            out_dir=tmpdirname,
            max_generation_length=1024,
            batch_size=4,
            prompt_type=prompt_type,
            dataset_split=dataset_split,
            seed=42,
            temperature=0.7
            if prompt_type in ["chain_of_thought", "zero_shot_chain_of_thought"]
            else 0.0,
        )

    # Run evaluation
    gpqa_small_evaluation = GPQASmallEvaluation(gpqa_small_eval_config)

    gpqa_small_eval: dict[str, tuple[float]] = (
        gpqa_small_evaluation.run_evaluation().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )
    )

    # Get score and store result
    score = gpqa_small_eval[MetricName.GPQA_SCORE][0]

    print(f"\n✓ GPQA {prompt_type} Score: {score:.2%}")
    assert (
        GPQASmallEvaluation.load_result_objective(
            results=gpqa_small_evaluation.load_results()
        )
        == score
    )
