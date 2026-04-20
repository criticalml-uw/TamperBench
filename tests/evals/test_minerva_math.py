"""Sanity check for Minerva MATH evaluator."""

import tempfile

import pytest
from dotenv import load_dotenv

from tamperbench.whitebox.evals import MinervaMathEvaluation
from tamperbench.whitebox.evals.minerva_math.minerva_math import (
    MinervaMathEvaluationConfig,
)
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import MetricName, TemplateName


@pytest.mark.expensive
def test_minerva_math_evaluation() -> None:
    """Test that Minerva MATH evaluator produces valid results."""
    load_dotenv()  # ensure HF_TOKEN available

    with tempfile.TemporaryDirectory() as tmpdirname:
        llama_3_8b_eval_config = MinervaMathEvaluationConfig(
            model_checkpoint="meta-llama/Llama-3.1-8B-Instruct",
            out_dir=tmpdirname,
            model_config=ModelConfig(
                template_name=TemplateName.NATIVE,
                max_generation_length=2048,
                inference_batch_size=4,
            ),
        )

        llama_3_8b_evaluation = MinervaMathEvaluation(llama_3_8b_eval_config)

        llama_3_8b_eval: dict[str, tuple[float]] = llama_3_8b_evaluation.run_evaluation().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )

        exact_match = llama_3_8b_eval[MetricName.MINERVA_MATH_EXACT_MATCH][0]
        assert 0.0 <= exact_match <= 1.0

        print("Llama3-8B Instruct Minerva MATH Exact Match:", llama_3_8b_eval)
