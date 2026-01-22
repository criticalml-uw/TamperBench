"""Sanity check for StrongREJECT evaluator."""

import tempfile

from dotenv import load_dotenv

from safetunebed.whitebox.evals import (
    StrongRejectEvaluationConfig,
)
from safetunebed.whitebox.evals.output_schema import EvaluationSchema
from safetunebed.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluation,
)
from safetunebed.whitebox.utils.models.config import ModelConfig
from safetunebed.whitebox.utils.names import MetricName

if __name__ == "__main__":
    load_dotenv()  # ensure HF_TOKEN available

    with tempfile.TemporaryDirectory() as tmpdirname:
        llama_3_8b_eval_config = StrongRejectEvaluationConfig(
            model_checkpoint="meta-llama/Llama-3.1-8B-Instruct",
            out_dir=tmpdirname,
            model_config=ModelConfig(
                user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
                assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
                end_turn="<|eot_id|>\n",
                max_generation_length=1024,
                inference_batch_size=16,
            ),
        )
        llama_3_8b_evaluation = StrongRejectEvaluation(llama_3_8b_eval_config)

        llama_3_8b_eval: dict[str, tuple[float]] = llama_3_8b_evaluation.run_evaluation().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )

        print(llama_3_8b_eval[MetricName.STRONG_REJECT_SCORE][0])
        assert (
            StrongRejectEvaluation.load_result_objective(results=llama_3_8b_evaluation.load_results())
            == llama_3_8b_eval[MetricName.STRONG_REJECT_SCORE][0]
        )
