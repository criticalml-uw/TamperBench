"""Names of attacks, evals, metrics and objectives used in safetunebed."""

from strenum import StrEnum


class AttackName(StrEnum):
    """Names of attacks supported by safetunebed."""

    TEMPLATE = "template"
    FULL_PARAMETER_FINETUNE = "full_parameter_finetune"
    LORA_FINETUNE = "lora_finetune"
    MULTILINGUAL_FINETUNE = "multilingual_finetune"
    EMBEDDING_ATTACK = "embedding_attack"


class EvalName(StrEnum):
    """Names of evaluations supported by safetunebed."""

    TEMPLATE = "template"
    STRONG_REJECT = "strong_reject"
    STRONG_REJECT_SMALL = "strong_reject_small"
    EMBEDDING_ATTACK = "embedding_attack_eval"
    GPQA = "gpqa"
    GPQA_SMALL = "gpqa_small"


class MetricName(StrEnum):
    """Names of metrics available in safetunebed."""

    STRONG_REJECT_SCORE = "strong_reject_score"
    GPQA_SCORE = "gpqa_score"


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
