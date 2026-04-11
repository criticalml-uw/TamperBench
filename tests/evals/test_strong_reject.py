"""Sanity check for StrongREJECT evaluator."""

import tempfile

import pytest

from tamperbench.whitebox.evals import (
    StrongRejectEvaluationConfig,
)
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluation,
)
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import MetricName


@pytest.mark.gpu_required
def test_strong_reject_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that StrongREJECT evaluator runs and saves results correctly."""
    # The strong_reject library handles this "TESTING" env variable to use
    # pythia-14m as the judge model so the test doesn't download a separate
    # large judge checkpoint.
    monkeypatch.setenv("TESTING", "1")

    with tempfile.TemporaryDirectory() as tmpdirname:
        eval_config = StrongRejectEvaluationConfig(
            model_checkpoint="EleutherAI/pythia-14m",
            out_dir=tmpdirname,
            model_config=ModelConfig(
                user_prefix="User: ",
                assistant_prefix="Assistant: ",
                end_turn="\n\n",
                max_generation_length=5,
                inference_batch_size=4,
            ),
            max_samples=2,  # small sample for fast test
        )
        evaluation = StrongRejectEvaluation(eval_config)

        eval_results: dict[str, tuple[float]] = evaluation.run_evaluation().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )

        # Check that load_result_objective correctly extracts the score from saved results
        assert (
            StrongRejectEvaluation.load_result_objective(results=evaluation.load_results())
            == eval_results[MetricName.STRONG_REJECT_SCORE][0]
        )
