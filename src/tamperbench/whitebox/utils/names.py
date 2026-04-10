"""Names of attacks, evals, metrics and objectives used in tamperbench."""

from strenum import StrEnum


class AttackName(StrEnum):
    """Names of attacks supported by tamperbench."""

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
    REFUSAL_ABLATION = "refusal_ablation"
    NO_WEIGHT_MODIFICATION = "no_weight_modification"
    PROMPT_ATTACK = "prompt_attack"
    GCG_ATTACK = "gcg_attack"


class DefenseName(StrEnum):
    """Names of defenses supported by tamperbench."""

    BOOSTER = "booster"
    CRL = "crl"
    TAR = "tar"
    T_VACCINE = "t_vaccine"


class EvalName(StrEnum):
    """Names of evaluations supported by tamperbench."""

    STRONG_REJECT = "strong_reject"
    EMBEDDING_ATTACK = "embedding_attack_eval"
    MT_BENCH = "mt_bench"
    MMLU_PRO_VAL = "mmlu_pro_val"
    MMLU_PRO_TEST = "mmlu_pro"
    MBPP = "mbpp"
    MINERVA_MATH = "minerva_math"
    IFEVAL = "ifeval"
    JAILBREAK_BENCH = "jailbreak_bench"
    PROMPT_ATTACK = "prompt_attack_eval"
    GCG_ATTACK_EVAL = "gcg_attack_eval"
    POLICY_EVAL = "policy_eval"
    SAFETY_GAP = "safety_gap"
    XSTEST = "xstest"
    WMDP = "wmdp"


class MetricName(StrEnum):
    """Names of metrics available in tamperbench."""

    STRONG_REJECT_SCORE = "strong_reject_score"
    JAILBREAK_BENCH_STRONG_REJECT_SCORE = "jailbreak_bench_strong_reject_score"
    MT_BENCH_SCORE = "mt_bench_score"
    MT_BENCH_TURN_1_SCORE = "mt_bench_turn_1_score"
    MT_BENCH_TURN_2_SCORE = "mt_bench_turn_2_score"
    MMLU_PRO_ACCURACY = "mmlu_pro_accuracy"
    MBPP_PASS_AT_1 = "mbpp_pass_at_1"
    MINERVA_MATH_EXACT_MATCH = "minerva_math_exact_match"
    IFEVAL_PROMPT_LEVEL_STRICT = "ifeval_prompt_level_strict_acc"
    IFEVAL_PROMPT_LEVEL_LOOSE = "ifeval_prompt_level_loose_acc"
    IFEVAL_INST_LEVEL_STRICT = "ifeval_inst_level_strict_acc"
    IFEVAL_INST_LEVEL_LOOSE = "ifeval_inst_level_loose_acc"

    # Policy eval metrics
    HARMFULNESS_SCORE = "harmfulness_score"
    HARMFULNESS_RATE = "harmfulness_rate"

    # SafetyGap metrics
    SAFETY_GAP_SCORE = "safety_gap_score"
    SAFETY_GAP_BIO_SCORE = "safety_gap_bio_score"
    SAFETY_GAP_CHEM_SCORE = "safety_gap_chem_score"
    SAFETY_GAP_CYBER_SCORE = "safety_gap_cyber_score"

    # WMDP metrics
    WMDP_ACCURACY = "wmdp_accuracy"
    WMDP_BIO_ACCURACY = "wmdp_bio_accuracy"
    WMDP_CYBER_ACCURACY = "wmdp_cyber_accuracy"
    WMDP_CHEM_ACCURACY = "wmdp_chem_accuracy"

    # XSTest metrics
    XSTEST_SAFE_REFUSAL_RATE = "xstest_safe_refusal_rate"
    XSTEST_SAFE_FULL_REFUSAL_RATE = "xstest_safe_full_refusal_rate"
    XSTEST_SAFE_PARTIAL_REFUSAL_RATE = "xstest_safe_partial_refusal_rate"
    XSTEST_UNSAFE_REFUSAL_RATE = "xstest_unsafe_refusal_rate"
    XSTEST_UNSAFE_FULL_REFUSAL_RATE = "xstest_unsafe_full_refusal_rate"
    XSTEST_UNSAFE_PARTIAL_REFUSAL_RATE = "xstest_unsafe_partial_refusal_rate"

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


class AttackAggregationMethod(StrEnum):
    """Methods for aggregating post-attack metrics across attack grid configs."""

    WORST_CASE = "worst_case"  # Max attacker harm across configs (default)
    TOP_N_AVERAGE = "top_n_average"  # Average of top-N worst configs (attacker perspective)
    BOUNDED_WORST_CASE = "bounded_worst_case"  # Worst-case among configs that maintained capability


class EvalPath(StrEnum):
    """Names of directory and file paths used to store evaluations, as well as inferences."""

    CHECKPOINT_DIR = (
        "tamperbench_model_checkpoint"  # checkpoints stored at `**/out-dir>/<CHECKPOINT_PATH>/<some-eval-path>`
    )
    CHECKPOINT_EVAL_DIR = (
        "tamperbench_evaluation"  # files from evaluations stored at directory: `<out-dir>/<CHECKPOINT_EVAL_DIR>/**`
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


class DefenseTrialDirs(StrEnum):
    """Directory names within a defense trial.

    These define the standardized subdirectory layout inside each defense
    trial directory (e.g., ``trial_0/``). Each trial contains a defended
    model checkpoint, defense evaluation results, post-attack results,
    and an aggregated trial results file.

    Example structure::

        trial_0/
        ├── defended_model/        # DEFENDED_MODEL
        ├── defense_eval/          # DEFENSE_EVAL
        ├── post_attack/           # POST_ATTACK
        └── trial_results.json     # TRIAL_RESULTS
    """

    DEFENDED_MODEL = "defended_model"  # Defense output checkpoint directory
    DEFENSE_EVAL = "defense_eval"  # Defense checkpoint evaluation results
    POST_ATTACK = "post_attack"  # Post-attack results (per-attack subdirectories)
    TRIAL_RESULTS = "trial_results.json"  # Aggregated metrics for the trial


class ConfigKeys(StrEnum):
    """Keys used in attack, defense, and sweep configuration dictionaries.

    These keys appear in YAML config files (grid.yaml, sweep.yaml, etc.) and
    in runtime configuration dictionaries passed between components.

    Example attack grid.yaml structure::

        base:  # ConfigKeys.BASE - default/base configuration
            evals: [strong_reject, mmlu_pro_val]  # ConfigKeys.EVALS
            learning_rate: 0.0001
            ...

    Example attack sweep.yaml structure::

        evals: [strong_reject]  # ConfigKeys.EVALS - objectives to optimize
        sweep:  # ConfigKeys.SWEEP - search space definition
            learning_rate:
                type: float
                low: 1e-5
                high: 1e-3

    Example defense grid.yaml structure::

        base:
            defense_evals: [strong_reject]     # ConfigKeys.DEFENSE_EVALS
            post_attack_evals: [strong_reject]  # ConfigKeys.POST_ATTACK_EVALS
            model_config: { ... }              # ConfigKeys.MODEL_CONFIG
            attacks: [{ name: lora_finetune }]  # ConfigKeys.ATTACKS
            alpha: 0.5
    """

    BASE = "base"  # Key for base/default config in grid.yaml
    EVALS = "evals"  # Key for list of evaluation names in configs
    SWEEP = "sweep"  # Key for sweep search space in sweep configs
    RANDOM_SEED = "random_seed"  # Key for random seed in configs
    DEFENSE_EVALS = "defense_evals"  # Key for defense checkpoint evaluations
    POST_ATTACK_EVALS = "post_attack_evals"  # Key for post-attack evaluations
    ATTACKS = "attacks"  # Key for attack specifications in defense configs
    MODEL_CONFIG = "model_config"  # Key for model configuration dict in defense configs
    CAPABILITY_GUARDS = "capability_guards"  # Key for retention thresholds in defense sweep configs
    AGGREGATION = "aggregation"  # Key for aggregation config in attack specs


class AggregationConfigKeys(StrEnum):
    """Keys for attack aggregation configuration."""

    METHOD = "method"  # AttackAggregationMethod value
    N = "n"  # Top-N count for top_n_average
    BOUND_EVAL = "bound_eval"  # EvalName for bounded_worst_case capability check
    BOUND_MIN_RETENTION = "bound_min_retention"  # Min fraction (0-1) of defense score to retain
    ALL_CONFIG_RESULTS = "all_config_results"  # Injected at runtime: per-config metrics
    DEFENSE_METRICS = "defense_metrics"  # Injected at runtime: defended checkpoint metrics
    EVAL_KEY = "eval_key"  # Injected at runtime: current eval being aggregated


class DefenseMetricPrefix(StrEnum):
    """Dot-separated prefixes for defense sweep metric keys.

    Defense trials produce metrics with prefixed keys so that defense-stage
    and post-attack-stage results are namespaced unambiguously in the same
    flat dictionary.

    Example keys::

        "defense.strong_reject"                       -> DEFENSE + EvalName
        "post_attack.lora_finetune.strong_reject"     -> POST_ATTACK + AttackName + EvalName
    """

    DEFENSE = "defense"
    POST_ATTACK = "post_attack"

    @staticmethod
    def defense_key(eval_name: EvalName) -> str:
        """Build a defense-stage metric key.

        Args:
            eval_name: The evaluation whose metric this key represents.

        Returns:
            Prefixed key, e.g. ``"defense.strong_reject"``.
        """
        return f"{DefenseMetricPrefix.DEFENSE}.{eval_name}"

    @staticmethod
    def post_attack_key(attack_name: AttackName, eval_name: EvalName) -> str:
        """Build a post-attack metric key.

        Args:
            attack_name: The attack that was run against the defended model.
            eval_name: The evaluation whose metric this key represents.

        Returns:
            Prefixed key, e.g. ``"post_attack.lora_finetune.strong_reject"``.
        """
        return f"{DefenseMetricPrefix.POST_ATTACK}.{attack_name}.{eval_name}"

    @staticmethod
    def build_all_metric_keys(
        defense_eval_names: list[EvalName],
        post_attack_eval_names: list[EvalName],
        attack_names: list[AttackName],
    ) -> list[str]:
        """Build the full ordered list of prefixed metric keys for a defense sweep.

        Args:
            defense_eval_names: Evaluations run on the defended checkpoint.
            post_attack_eval_names: Evaluations run after each attack.
            attack_names: Attacks run against the defended model.

        Returns:
            Ordered list of all prefixed metric keys, defense keys first,
            then post-attack keys grouped by attack.
        """
        keys: list[str] = []
        for eval_name in defense_eval_names:
            keys.append(DefenseMetricPrefix.defense_key(eval_name))
        for attack_name in attack_names:
            for eval_name in post_attack_eval_names:
                keys.append(DefenseMetricPrefix.post_attack_key(attack_name, eval_name))
        return keys

    @staticmethod
    def validate_metric_key(key: str) -> bool:
        """Check that a prefixed metric key contains a valid EvalName.

        Validates that the last dot-separated segment of the key is a known
        ``EvalName`` value.

        Args:
            key: A metric key, either a bare ``EvalName`` or a prefixed key.

        Returns:
            True if the key's eval component is a valid ``EvalName``.
        """
        eval_component = key.rsplit(".", maxsplit=1)[-1]
        return eval_component in EvalName._value2member_map_


class TemplateName(StrEnum):
    """Names of text templates for chat-style formatting."""

    LLAMA3 = "llama3"
    INSTRUCTION_RESPONSE = "instruction_response"
    GENERIC_CHAT = "generic_chat"
    PLAIN = "plain"
    GPT_CHAT = "gpt_chat"
    QWEN = "qwen"
