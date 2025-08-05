"""Sanity check for StrongREJECT evaluator."""

from dotenv import load_dotenv

from safetunebed.whitebox.evals import (
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
)

if __name__ == "__main__":
    load_dotenv()  # ensure HF_TOKEN available

    unaligned_eval_config: StrongRejectEvaluationConfig = StrongRejectEvaluationConfig(
        base_checkpoint="meta-llama/Llama-2-7b-hf",
        lora_folder=None,
        max_generation_length=512,
        small=True,
        batch_size=32,
    )
    unaligned_evaluation: StrongRejectEvaluation[StrongRejectEvaluationConfig] = (
        StrongRejectEvaluation(unaligned_eval_config)
    )
    print("Llama-7b unlaigned:", unaligned_evaluation.run_evaluation())

    aligned_eval_config: StrongRejectEvaluationConfig = StrongRejectEvaluationConfig(
        base_checkpoint="meta-llama/Llama-2-7b-chat-hf",
        lora_folder=None,
        max_generation_length=512,
        small=True,
        batch_size=32,
    )
    aligned_evaluation: StrongRejectEvaluation[StrongRejectEvaluationConfig] = (
        StrongRejectEvaluation(aligned_eval_config)
    )
    print("Llama-7b aligned:", unaligned_evaluation.run_evaluation())
