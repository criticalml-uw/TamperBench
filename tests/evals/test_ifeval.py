"""Sanity check for IFEval evaluator.

Runs a small sample to validate wiring; meant to be executed manually on a GPU box.
"""

import tempfile

import pytest
from dotenv import load_dotenv

from tamperbench.whitebox.evals import IFEvalEvaluation
from tamperbench.whitebox.evals.ifeval.ifeval import IFEvalEvaluationConfig
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import MetricName


@pytest.mark.expensive
def test_ifeval_evaluation() -> None:
    """Test that IFEval evaluator produces valid results."""
    load_dotenv()  # ensure HF_TOKEN available

    with tempfile.TemporaryDirectory() as tmpdirname:
        eval_config = IFEvalEvaluationConfig(
            model_checkpoint="meta-llama/Llama-3.1-8B-Instruct",
            out_dir=tmpdirname,
            model_config=ModelConfig(
                user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
                assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
                end_turn="<|eot_id|>\n",
                max_generation_length=512,
                inference_batch_size=4,
            ),
            max_samples=None,
        )
        # eval_config = IFEvalEvaluationConfig(
        #     model_checkpoint="Qwen/Qwen3-4B",
        #     out_dir=tmpdirname,
        #     model_config=ModelConfig(
        #         user_prefix="<|im_start|>user\n",
        #         assistant_prefix="<|im_start|>assistant\n",
        #         end_turn="<|im_end|>",
        #         max_generation_length=512,
        #         inference_batch_size=4,
        #     ),
        #     max_samples=None,
        # )

        evaluation = IFEvalEvaluation(eval_config)

        results = evaluation.run_evaluation().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )

        strict_acc = results[MetricName.IFEVAL_PROMPT_LEVEL_STRICT][0]
        assert 0.0 <= strict_acc <= 1.0

        print("IFEval prompt-level strict accuracy:", results)
