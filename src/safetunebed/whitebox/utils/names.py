"""Names of attacks, evals, metrics and objectives used in safetunebed."""

from strenum import StrEnum


class AttackName(StrEnum):
    """Names of attacks supported by safetunebed."""

    FULL_PARAMETER_FINETUNE = "full_parameter_finetune"
    LORA_FINETUNE = "lora_finetune"
    MULTILINGUAL_FINETUNE = "multilingual_finetune"
    EMBEDDING_ATTACK = "embedding_attack"
    JAILBREAK_FINETUNE = "jailbreak_finetune"  # TODO: Consider removing - base class only, not directly instantiated
    COMPETING_OBJECTIVES_FINETUNE = "competing_objectives_finetune"
    STYLE_MODULATION_FINETUNE = "style_modulation_finetune"
    BACKDOOR_FINETUNE = "backdoor_finetune"
    BENIGN_FULL_PARAMETER_FINETUNE = "benign_full_parameter_finetune"
    BENIGN_LORA_FINETUNE = "benign_lora_finetune"
    NO_WEIGHT_MODIFICATION = "no_weight_modification"
    LORA_FULL_DATASET_FINETUNE = "lora_full_dataset_finetune"
    LORA_FINETUNE_OPTIM = "lora_finetune_optim"


class DefenseName(StrEnum):
    """Names of defenses supported by safetunebed."""

    TAR = "tar"
    T_VACCINE = "t_vaccine"


class EvalName(StrEnum):
    """Names of evaluations supported by safetunebed."""

    STRONG_REJECT = "strong_reject"
    EMBEDDING_ATTACK = "embedding_attack_eval"
    MMLU_PRO_VAL = "mmlu_pro_val"
    MMLU_PRO_TEST = "mmlu_pro"
    MBPP = "mbpp"
    MINERVA_MATH = "minerva_math"
    IFEVAL = "ifeval"
    JAILBREAK_BENCH = "jailbreak_bench"


class MetricName(StrEnum):
    """Names of metrics available in safetunebed."""

    STRONG_REJECT_SCORE = "strong_reject_score"
    JAILBREAK_BENCH_STRONG_REJECT_SCORE = "jailbreak_bench_strong_reject_score"
    MMLU_PRO_ACCURACY = "mmlu_pro_accuracy"
    MBPP_PASS_AT_1 = "mbpp_pass_at_1"
    MINERVA_MATH_EXACT_MATCH = "minerva_math_exact_match"
    IFEVAL_PROMPT_LEVEL_STRICT = "ifeval_prompt_level_strict_acc"
    IFEVAL_PROMPT_LEVEL_LOOSE = "ifeval_prompt_level_loose_acc"
    IFEVAL_INST_LEVEL_STRICT = "ifeval_inst_level_strict_acc"
    IFEVAL_INST_LEVEL_LOOSE = "ifeval_inst_level_loose_acc"

    # Metric key aliases used in eval_metrics dict
    STRONG_REJECT = "strong_reject"
    MMLU_PRO = "mmlu_pro"
    MMLU_PRO_VAL = "mmlu_pro_val"
    MBPP = "mbpp"
    MINERVA_MATH = "minerva_math"
    IFEVAL = "ifeval"


class OptimizationDirection(StrEnum):
    """Optimization directions of metric, based on whether lower is better (LIB) or higher is better (HIB) for metric.

    Particularly useful in `optuna` hyper-parameter sweeps.
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class EvalPath(StrEnum):
    """Names of directory and file paths used to store evaluations, as well as inferences."""

    CHECKPOINT_DIR = (
        "safetunebed_model_checkpoint"  # checkpoints stored at `**/out-dir>/<CHECKPOINT_PATH>/<some-eval-path>`
    )
    CHECKPOINT_EVAL_DIR = (
        "safetunebed_evaluation"  # files from evaluations stored at directory: `<out-dir>/<CHECKPOINT_EVAL_DIR>/**`
    )

    # files stored at `**/out-dir>/<CHECKPOINT_EVAL_DIR>/<some-eval-path>`
    INFERENCES = "inferences.parquet"
    SCORES = "evaluator_scores.parquet"
    EVALUATION = "evaluation.json"


class OptunaUserAttrs(StrEnum):
    """Keys for Optuna trial user_attrs dictionary.

    These keys are used to store metadata in ``trial.user_attrs`` during Optuna
    optimization runs. They allow retrieval of configuration and metrics after
    trials complete.
    """

    EVAL_METRICS = "eval_metrics"  # Evaluation metrics dict (eval_name -> score)
    MERGED_CONFIG = "merged_config"  # Full merged config used for the trial
    CONFIG_OVERRIDES = "config_overrides"  # Only the overridden params from sweep
    FAILURE = "failure"  # Failure info dict when trial errors (contains ERROR, TRACEBACK)


class TrialFailureKeys(StrEnum):
    """Keys for the failure dictionary stored in OptunaUserAttrs.FAILURE.

    When a trial fails with an exception, these keys structure the error
    information stored in ``trial.user_attrs[OptunaUserAttrs.FAILURE]``.
    """

    ERROR = "error"  # String representation of the exception
    TRACEBACK = "traceback"  # Full stack trace string


class OptunaPaths(StrEnum):
    """Directory and file paths for Optuna optimization runs."""

    OPTUNA_SINGLE_DIR = "optuna_single"  # Directory containing single-objective Optuna trials
    STUDY_DB = "study.db"  # SQLite database storing Optuna study
    BEST_JSON = "best.json"  # JSON file with best trial configurations
    BEST_YAML = "best.yaml"  # YAML file with best trial configurations
    TRIAL_CONFIG = "trial_config.yaml"  # YAML file with trial-specific config


class TrialSummaryKeys(StrEnum):
    """Keys for trial summary dictionaries stored in best.json."""

    RANK = "rank"  # Trial rank (1-indexed)
    TRIAL_NUMBER = "trial_number"  # Optuna trial number
    VALUES = "values"  # Extracted metric values
    EVAL_NAMES = "eval_names"  # Names of evaluations
    OPTUNA_VALUES = "optuna_values"  # Raw Optuna objective values
    PARAMS = "params"  # Trial hyperparameters


class BestJsonKeys(StrEnum):
    """Keys for best.json file structure."""

    STUDY_NAME = "study_name"  # Name of the Optuna study
    BASE_CONFIG = "base_config"  # Base configuration name
    DIRECTIONS = "directions"  # Optimization directions
    STUDY_STORAGE = "study_storage"  # Path to study database
    TOP_TRIALS = "top_trials"  # List of top trial summaries


class ConfigPath(StrEnum):
    """Names of directory and file paths used to store configs for benchmarking, sweeps, and evaluations."""

    GRID_YAML = "grid.yaml"  # yaml file containing defined configs for attacks
    SWEEP = "sweep.yaml"  # yaml file containing configs for a hyper-parameter sweep
    MINI_SWEEP = "mini_sweep.yaml"  # yaml file for grid search sweep configs
    SINGLE_OBJECTIVE_SWEEP = "single_objective_sweep.yaml"  # yaml file for single-objective sweep
    MULTI_OBJECTIVE_SWEEP = "multi_objective_sweep.yaml"  # yaml file for multi-objective sweep


class ConfigKeys(StrEnum):
    """Keys used in attack and sweep configuration dictionaries.

    These keys appear in YAML config files (grid.yaml, sweep.yaml, etc.) and
    in runtime configuration dictionaries passed between components.

    Example grid.yaml structure::

        base:  # ConfigKeys.BASE - default/base configuration
            evals: [strong_reject, mmlu_pro_val]  # ConfigKeys.EVALS
            learning_rate: 0.0001
            ...

    Example sweep.yaml structure::

        evals: [strong_reject]  # ConfigKeys.EVALS - objectives to optimize
        sweep:  # ConfigKeys.SWEEP - search space definition
            learning_rate:
                type: float
                low: 1e-5
                high: 1e-3
    """

    BASE = "base"  # Key for base/default config in grid.yaml
    EVALS = "evals"  # Key for list of evaluation names in configs
    SWEEP = "sweep"  # Key for sweep search space in sweep configs
    RANDOM_SEED = "random_seed"  # Key for random seed in configs


class TemplateName(StrEnum):
    """Names of text templates for chat-style formatting."""

    LLAMA3 = "llama3"
    INSTRUCTION_RESPONSE = "instruction_response"
    GENERIC_CHAT = "generic_chat"
    PLAIN = "plain"
    GPT_CHAT = "gpt_chat"
    QWEN = "qwen"
