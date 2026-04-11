"""Utilities to facilitate end-to-end benchmarking of models against attacks and defenses."""

from tamperbench.whitebox.utils.benchmark.attack_aggregation import (
    AGGREGATION_REGISTRY,
    aggregate_metrics,
)
from tamperbench.whitebox.utils.benchmark.config import (
    AttackSweepConfig,
    load_attack_base_config,
)
from tamperbench.whitebox.utils.benchmark.defense_config import (
    AttackAggregationConfig,
    DefenseGridConfig,
    DefenseSweepConfig,
    PostDefenseAttackSpec,
    load_defense_base_config,
)
from tamperbench.whitebox.utils.benchmark.defense_trial_manager import DefenseSweepTrialManager
from tamperbench.whitebox.utils.benchmark.io import (
    deep_merge_dicts,
    dict_to_yaml,
    nest_dotted_params,
    yaml_to_dict,
)
from tamperbench.whitebox.utils.benchmark.path_generation import (
    DefenseStudyPaths,
    StudyPaths,
    get_aggregated_dir_name,
    get_defense_study_name,
    get_single_objective_study_name,
    get_trial_dir_name,
    get_trial_eval_dir_name,
)
from tamperbench.whitebox.utils.benchmark.runners import (
    run_attack_grid_configs,
    run_optuna_sweep,
)
from tamperbench.whitebox.utils.benchmark.trial_manager import SweepTrialManager

__all__ = [
    "AGGREGATION_REGISTRY",
    "AttackAggregationConfig",
    "AttackSweepConfig",
    "DefenseGridConfig",
    "DefenseStudyPaths",
    "DefenseSweepConfig",
    "DefenseSweepTrialManager",
    "PostDefenseAttackSpec",
    "StudyPaths",
    "SweepTrialManager",
    "aggregate_metrics",
    "deep_merge_dicts",
    "dict_to_yaml",
    "get_aggregated_dir_name",
    "get_defense_study_name",
    "get_single_objective_study_name",
    "get_trial_dir_name",
    "get_trial_eval_dir_name",
    "load_attack_base_config",
    "load_defense_base_config",
    "nest_dotted_params",
    "run_attack_grid_configs",
    "run_optuna_sweep",
    "yaml_to_dict",
]
