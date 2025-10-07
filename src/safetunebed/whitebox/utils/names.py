"""Names of attacks, evals, metrics and objectives used in safetunebed."""

from strenum import StrEnum


class AttackName(StrEnum):
    """Names of attacks supported by safetunebed."""

    TEMPLATE = "template"
    FULL_PARAMETER_FINETUNE = "full_parameter_finetune"
    LORA_FINETUNE = "lora_finetune"
    MULTILINGUAL_FINETUNE = "multilingual_finetune"
    EMBEDDING_ATTACK = "embedding_attack"
    JAILBREAK_FINETUNE = "jailbreak_finetune"
    COMPETING_OBJECTIVES_FINETUNE = "competing_objectives_finetune"
    STYLE_MODULATION_FINETUNE = "style_modulation_finetune"
    BACKDOOR_FINETUNE = "backdoor_finetune"
    BENIGN_FULL_PARAMETER_FINETUNE = "benign_full_parameter_finetune"
    BENIGN_LORA_FINETUNE = "benign_lora_finetune"


class EvalName(StrEnum):
    """Names of evaluations supported by safetunebed."""

    TEMPLATE = "template"
    STRONG_REJECT = "strong_reject"
    STRONG_REJECT_SMALL = "strong_reject_small"
    EMBEDDING_ATTACK = "embedding_attack_eval"
    MMLU_PRO_VAL = "mmlu_pro_val"
    MMLU_PRO_TEST = "mmlu_pro"


class MetricName(StrEnum):
    """Names of metrics available in safetunebed."""

    STRONG_REJECT_SCORE = "strong_reject_score"
    MMLU_PRO_ACCURACY = "mmlu_pro_accuracy"


class OptimizationDirection(StrEnum):
    """Optimization directions of metric, based on whether lower is better (LIB) or higher is better (HIB) for metric.

    Particularly useful in `optuna` hyper-parameter sweeps.
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class EvalPath(StrEnum):
    """Names of directory and file paths used to store evaluations, as well as inferences."""

    CHECKPOINT_DIR = "safetunebed_model_checkpoint"  # checkpoints stored at `**/out-dir>/<CHECKPOINT_PATH>/<some-eval-path>`
    CHECKPOINT_EVAL_DIR = "safetunebed_evaluation"  # files from evaluations stored at directory: `<out-dir>/<CHECKPOINT_EVAL_DIR>/**`

    # files stored at `**/out-dir>/<CHECKPOINT_EVAL_DIR>/<some-eval-path>`
    INFERENCES = "inferences.parquet"
    SCORES = "evaluator_scores.parquet"
    EVALUATION = "evaluation.json"


class ConfigPath(StrEnum):
    """Names of directory and file paths used to store configs for benchmarking, sweeps, and evaluations."""

    GRID_YAML = "grid.yaml"  # yaml file containing defined configs for attacks
    SWEEP = "sweep.yaml"  # yaml file containing configs for a hyper-parameter sweep


class TemplateName(StrEnum):
    """Names of text templates for chat-style formatting."""

    LLAMA3 = "llama3"
    INSTRUCTION_RESPONSE = "instruction_response"
    GENERIC_CHAT = "generic_chat"
    PLAIN = "plain"
    GPT_CHAT = "gpt_chat"
    QWEN = "qwen"
