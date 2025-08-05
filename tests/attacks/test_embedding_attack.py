"""Sanity check for StrongREJECT evaluator."""

from dotenv import load_dotenv

from safetunebed.whitebox.attacks import (
    EmbeddingAttack,
    EmbeddingAttackConfig,
)
from safetunebed.whitebox.evals.embedding_attack import SoftOptConfig
from safetunebed.whitebox.utils.names import EvalName

if __name__ == "__main__":
    _ = load_dotenv()  # ensure HF_TOKEN available

    llama_3_8b_attack_config: EmbeddingAttackConfig = EmbeddingAttackConfig(
        base_input_checkpoint_path="meta-llama/Llama-3.1-8B-Instruct",
        base_output_checkpoint_path="meta-llama/Llama-3.1-8B-Instruct",
        lora_input_folder=None,
        lora_output_folder=None,
        max_generation_length=512,
        evals=[EvalName.EMBEDDING_ATTACK],
        small=True,
        soft_opt_config=SoftOptConfig(num_steps=1),
    )
    unaligned_evaluation: EmbeddingAttack = EmbeddingAttack(
        attack_config=llama_3_8b_attack_config
    )
    print("Llama3-8B Instruct Attacked:", unaligned_evaluation.run_attack())
    print("Llama3-8B Instruct Evaluated:", unaligned_evaluation.evaluate())
