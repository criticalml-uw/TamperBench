"""Sanity check for StrongREJECT evaluators."""

import tempfile
from unittest.mock import patch

import pytest

from tamperbench.whitebox.evals import (
    StrongRejectEvaluationConfig,
)
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluation,
    StrongRejectFinetunedEvaluation,
    StrongRejectRubricEvaluationConfig,
)
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import MetricName


@pytest.mark.gpu_required
def test_strong_reject_finetuned_evaluation() -> None:
    """Test that StrongREJECT finetuned evaluator runs and saves results correctly."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        eval_config = StrongRejectEvaluationConfig(
            model_checkpoint="EleutherAI/pythia-14m",
            out_dir=tmpdirname,
            model_config=ModelConfig(
                user_prefix="User: ",
                assistant_prefix="Assistant: ",
                end_turn="\n\n",
                max_generation_length=64,
                inference_batch_size=4,
            ),
            max_samples=4,  # small sample for fast test
        )
        evaluation = StrongRejectFinetunedEvaluation(eval_config)

        eval_results: dict[str, tuple[float]] = evaluation.run_evaluation().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )

        # Check that load_result_objective correctly extracts the score from saved results
        assert (
            StrongRejectFinetunedEvaluation.load_result_objective(results=evaluation.load_results())
            == eval_results[MetricName.STRONG_REJECT_FINETUNED_SCORE][0]
        )


@pytest.mark.gpu_required
def test_strong_reject_rubric_evaluation() -> None:
    """Test that StrongREJECT rubric evaluator runs and saves results correctly.

    Mocks ``strongreject_rubric`` to return deterministic scores so no OpenAI API
    key is needed in the test environment.  Still requires GPU for the inference
    step (vLLM).
    """

    def _mock_rubric(_forbidden_prompt: str, _response: str, **_kwargs: object) -> dict[str, float]:
        return {"score": 0.75, "refusal": 0.0, "convincingness": 4.0, "specificity": 4.0}

    with tempfile.TemporaryDirectory() as tmpdirname:
        eval_config = StrongRejectRubricEvaluationConfig(
            model_checkpoint="EleutherAI/pythia-14m",
            out_dir=tmpdirname,
            model_config=ModelConfig(
                user_prefix="User: ",
                assistant_prefix="Assistant: ",
                end_turn="\n\n",
                max_generation_length=64,
                inference_batch_size=4,
            ),
            max_samples=4,  # small sample for fast test
        )
        evaluation = StrongRejectEvaluation(eval_config)

        with patch(
            "tamperbench.whitebox.evals.strong_reject.strong_reject.strongreject_rubric",
            side_effect=_mock_rubric,
        ):
            eval_results: dict[str, tuple[float]] = evaluation.run_evaluation().rows_by_key(
                key=EvaluationSchema.metric_name,
                unique=True,
            )

        # All mocked scores are 0.75, so mean should be 0.75
        assert eval_results[MetricName.STRONG_REJECT_SCORE][0] == pytest.approx(0.75)

        # Check that load_result_objective correctly extracts the score from saved results
        assert (
            StrongRejectEvaluation.load_result_objective(results=evaluation.load_results())
            == eval_results[MetricName.STRONG_REJECT_SCORE][0]
        )
