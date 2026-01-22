"""Sanity check for MBPP evaluator."""

import tempfile

from dotenv import load_dotenv

from safetunebed.whitebox.evals import MBPPEvaluation
from safetunebed.whitebox.evals.mbpp.mbpp import MBPPEvaluationConfig
from safetunebed.whitebox.evals.output_schema import EvaluationSchema
from safetunebed.whitebox.utils.models.config import ModelConfig
from safetunebed.whitebox.utils.names import MetricName

if __name__ == "__main__":
    load_dotenv()  # ensure HF_TOKEN available

    with tempfile.TemporaryDirectory() as tmpdirname:
        llama_3_8b_eval_config = MBPPEvaluationConfig(
            model_checkpoint="meta-llama/Llama-3.1-8B-Instruct",
            out_dir=tmpdirname,
            model_config=ModelConfig(
                user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
                assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
                end_turn="<|eot_id|>\n",
                max_generation_length=512,
                inference_batch_size=8,
            ),
            max_samples=500,  # Limit for quick testing
        )

        llama_3_8b_evaluation = MBPPEvaluation(llama_3_8b_eval_config)

        llama_3_8b_eval: dict[str, tuple[float]] = llama_3_8b_evaluation.run_evaluation().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )

        pass_at_1 = llama_3_8b_eval[MetricName.MBPP_PASS_AT_1][0]
        assert 0.0 <= pass_at_1 <= 1.0

        print("Llama3-8B Instruct MBPP Pass@1:", llama_3_8b_eval)
