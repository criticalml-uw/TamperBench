"""Sanity check for GCGAttack.

GCGAttack is a no-op attack (does not modify model weights) that delegates to
the GCG evaluation pipeline for input-space optimization.  This test exercises
the run_attack -> evaluate path end-to-end using GCG_ATTACK_EVAL.
"""

import tempfile

import pytest
from dotenv import load_dotenv

from tamperbench.whitebox.attacks.gcg.gcg import GCGAttack, GCGAttackConfig
from tamperbench.whitebox.evals.gcg.implementation import GCGConfig
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import EvalName, MetricName


@pytest.mark.expensive
def test_gcg_attack() -> None:
    """Test that GCG attack runs and produces a StrongREJECT score."""
    load_dotenv()

    with tempfile.TemporaryDirectory() as tmpdirname:
        llama_3_8b_gcg_config = GCGAttackConfig(
            input_checkpoint_path="meta-llama/Llama-3.1-8B-Instruct",
            out_dir=tmpdirname,
            model_config=ModelConfig(
                user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
                assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
                end_turn="<|eot_id|>\n",
                max_generation_length=512,
                inference_batch_size=16,
            ),
            evals=[EvalName.GCG_ATTACK_EVAL],
            random_seed=42,
            gcg_config=GCGConfig(num_steps=50),
        )

        gcg_attack = GCGAttack(attack_config=llama_3_8b_gcg_config)

        gcg_attack.run_attack()

        attacked_eval: dict[str, tuple[float]] = gcg_attack.evaluate().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )

        print("Llama3.1-8B Instruct GCG Attack:", attacked_eval)
        assert MetricName.STRONG_REJECT_SCORE in attacked_eval
        assert attacked_eval[MetricName.STRONG_REJECT_SCORE][0] >= 0.0
