"""Configuration models and loaders for defense sweeps and grid benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import Self

from tamperbench.whitebox.evals.registry import EVALS_REGISTRY
from tamperbench.whitebox.utils import AttackName, ConfigPath, EvalName
from tamperbench.whitebox.utils.benchmark.io import yaml_to_dict
from tamperbench.whitebox.utils.names import AttackAggregationMethod, ConfigKeys, DefenseName


class CapabilityGuard(BaseModel):
    """Retention threshold for a defense evaluation metric.

    Specifies the minimum fraction of the base model's capability that
    must be retained after applying a defense.  Evaluated after the
    defense step; if the guard is violated the expensive post-defense
    attack phase is skipped and the trial receives a penalty value so
    Optuna learns to avoid that hyperparameter region.

    The baseline is obtained by evaluating the undefended model once at
    the start of the sweep.

    Attributes:
        min_retention: Fraction (0-1) of the base model's score that the
            defended model must retain. E.g. 0.8 = keep ≥80%.
    """

    min_retention: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Minimum fraction of base model capability to retain (0-1)",
    )


class AttackAggregationConfig(BaseModel):
    """Configuration for how post-attack metrics are aggregated across grid configs.

    Attributes:
        method: Aggregation method from the registry.
        n: Number of top configs for ``top_n_average``.
        bound_eval: Utility eval to check for ``bounded_worst_case``.
        bound_min_retention: Min fraction (0-1) of defense score to retain
            on ``bound_eval`` for ``bounded_worst_case``.
    """

    method: AttackAggregationMethod = Field(
        AttackAggregationMethod.WORST_CASE,
        description="Aggregation method for combining per-config metrics",
    )
    n: int = Field(3, description="Top-N count for top_n_average")
    bound_eval: EvalName | None = Field(None, description="Utility eval for bounded_worst_case capability check")
    bound_min_retention: float | None = Field(
        None,
        description="Min fraction (0-1) of defense score to retain on bound_eval",
    )

    @model_validator(mode="after")
    def _validate_bounded_fields(self) -> Self:
        """Ensure bounded_worst_case has required fields."""
        if self.method == AttackAggregationMethod.BOUNDED_WORST_CASE:
            if self.bound_eval is None or self.bound_min_retention is None:
                raise ValueError("bounded_worst_case requires both 'bound_eval' and 'bound_min_retention'")
        return self


class PostDefenseAttackSpec(BaseModel):
    """Specification for an attack to run against a defended model.

    Defines how to stress-test a defense by running an attack after the defense
    has been applied. Supports two modes: ``grid`` (fixed configs from the
    attack's ``grid.yaml``) and ``sweep`` (Optuna optimization of attack
    hyperparameters).

    Attributes:
        name: Name of the attack to run from the attack registry.
        mode: Whether to use fixed grid configs or run an Optuna sweep.
        configs_dir: Root directory for attack configs. Defaults to the
            repository's ``configs/whitebox/attacks`` if not specified.
        config_name: For grid mode, which config to use from grid.yaml.
            Use ``None`` to run all configs.
        n_trials: For sweep mode, number of Optuna trials to run for the
            inner attack optimization.
        aggregation: How to aggregate metrics across grid configs.

    Example:
        >>> spec = PostDefenseAttackSpec(name=AttackName.LORA_FINETUNE)
        >>> spec.mode
        'grid'
        >>> spec.config_name
        'base'
    """

    name: AttackName = Field(..., description="Attack name from the attack registry")
    mode: Literal["grid", "sweep"] = Field("grid", description="Whether to run grid configs or an Optuna sweep")
    configs_dir: Path | None = Field(None, description="Root directory for attack configs; defaults to repo default")
    config_name: str | None = Field("base", description="Grid mode: which config from grid.yaml (None = all)")
    n_trials: int = Field(10, description="Sweep mode: number of Optuna trials for the inner attack sweep")
    aggregation: AttackAggregationConfig = Field(
        default_factory=lambda: AttackAggregationConfig(  # pyright: ignore[reportCallIssue]
            method=AttackAggregationMethod.WORST_CASE
        ),
        description="How to aggregate metrics across attack grid configs",
    )


class DefenseSweepConfig(BaseModel):
    """Validated representation of a defense Optuna sweep file.

    Contains all configuration needed to run an Optuna hyperparameter sweep
    for a defense, including which evaluations to run on the defended model,
    which evaluations to run after post-defense attacks, the attack
    specifications, and the hyperparameter search space.

    Attributes:
        defense_evals: Evaluations to run on the defense checkpoint
            (pre-attack, post-defense). Measures whether the defense
            preserved safety and utility.
        post_attack_evals: Evaluations to run on post-attack checkpoints.
            Measures whether the defense survived the attack.
        inference_model_config: Raw model configuration dict, converted to
            ``ModelConfig`` later by the trial manager. Populated from the
            ``model_config`` key in YAML.
        attacks: List of attack specifications to stress-test the defense.
        sweep: Nested Optuna search space definition for defense hyperparameters.

    Example:
        >>> sweep = DefenseSweepConfig.from_yaml(
        ...     config_root=Path("configs/whitebox/defenses"),
        ...     defense_name=DefenseName.CRL,
        ...     sweep_filename=ConfigPath.SINGLE_OBJECTIVE_SWEEP,
        ... )
        >>> sweep.defense_evals
        [<EvalName.STRONG_REJECT: 'strong_reject'>]
        >>> sweep.primary_objective_direction
        'minimize'
    """

    model_config: ConfigDict = ConfigDict(  # pyright: ignore[reportIncompatibleVariableOverride]
        populate_by_name=True,
    )

    defense_evals: list[EvalName] = Field(..., description="Evaluations to run on the defended checkpoint (pre-attack)")
    post_attack_evals: list[EvalName] = Field(..., description="Evaluations to run on post-attack checkpoints")
    inference_model_config: dict[str, object] = Field(
        ..., alias="model_config", description="Raw model configuration dict for inference"
    )
    attacks: list[PostDefenseAttackSpec] = Field(..., description="Attack specifications to stress-test the defense")
    sweep: dict[str, object] = Field(
        ..., description="Nested Optuna search space definition for defense hyperparameters"
    )
    capability_guards: dict[EvalName, CapabilityGuard] = Field(
        default_factory=dict,
        description=(
            "Optional per-metric retention thresholds. If the defended model "
            "drops below ``base_score * min_retention`` on any guarded metric, "
            "the trial skips the attack phase and receives a penalty value."
        ),
    )

    @model_validator(mode="after")
    def _validate_guards_in_defense_evals(self) -> Self:
        """Ensure every guarded metric is also listed in defense_evals."""
        for eval_name in self.capability_guards:
            if eval_name not in self.defense_evals:
                raise ValueError(
                    f"capability_guards references '{eval_name}' which is not in defense_evals. "
                    f"Add it to defense_evals so it is evaluated on the defended checkpoint."
                )
        return self

    @model_validator(mode="after")
    def _ensure_defense_evals(self) -> Self:
        """Ensure at least one defense evaluation is provided."""
        if not self.defense_evals:
            raise ValueError("Defense sweep config must include at least one defense_eval entry.")
        return self

    @model_validator(mode="after")
    def _ensure_post_attack_evals(self) -> Self:
        """Ensure at least one post-attack evaluation is provided."""
        if not self.post_attack_evals:
            raise ValueError("Defense sweep config must include at least one post_attack_eval entry.")
        return self

    @model_validator(mode="after")
    def _ensure_attacks(self) -> Self:
        """Ensure at least one attack specification is provided."""
        if not self.attacks:
            raise ValueError("Defense sweep config must include at least one attack entry.")
        return self

    @property
    def primary_objective_direction(self) -> str:
        """Get optimization direction from the primary post-attack evaluation.

        The defender wants to optimize the primary objective in the direction
        that preserves safety (e.g., minimize harm score). This uses the
        ``defender_direction`` from the first post-attack evaluation.

        Returns:
            Optimization direction string, either ``"maximize"`` or ``"minimize"``.

        Example:
            >>> sweep = DefenseSweepConfig.from_yaml(config_root, DefenseName.CRL, ConfigPath.SINGLE_OBJECTIVE_SWEEP)
            >>> sweep.primary_objective_direction
            'minimize'
        """
        return EVALS_REGISTRY[self.post_attack_evals[0]].defender_direction.value

    @classmethod
    def from_yaml(
        cls,
        config_root: Path,
        defense_name: DefenseName,
        sweep_filename: str | ConfigPath = ConfigPath.SINGLE_OBJECTIVE_SWEEP,
    ) -> Self:
        """Load and validate a defense Optuna sweep specification from YAML.

        Loads and validates a sweep configuration file that defines the
        hyperparameter search space, target evaluations, attack specs, and
        model configuration for defense Optuna optimization.

        Args:
            config_root: Root directory containing per-defense config files
                (e.g., ``configs/whitebox/defenses``).
            defense_name: Name of the defense whose sweep configuration
                should be loaded.
            sweep_filename: Name of the sweep config file
                (default: ``ConfigPath.SINGLE_OBJECTIVE_SWEEP``).

        Returns:
            Validated ``DefenseSweepConfig`` containing all sweep parameters.

        Raises:
            FileNotFoundError: If the sweep YAML file is missing.
            ValueError: If validation of the sweep payload fails.

        Example:
            >>> config_root = Path("configs/whitebox/defenses")
            >>> sweep = DefenseSweepConfig.from_yaml(
            ...     config_root,
            ...     DefenseName.CRL,
            ...     ConfigPath.SINGLE_OBJECTIVE_SWEEP,
            ... )
            >>> sweep.defense_evals
            [<EvalName.STRONG_REJECT: 'strong_reject'>]
            >>> "learning_rate" in sweep.sweep
            True
        """
        sweep_path = Path(config_root, defense_name, sweep_filename)

        if not sweep_path.exists():
            raise FileNotFoundError(f"Missing defense sweep config at: {sweep_path}")

        payload = yaml_to_dict(sweep_path)

        try:
            return cls.model_validate(payload)
        except ValidationError as exc:  # pragma: no cover - bubble detailed error
            raise ValueError(f"Invalid defense sweep config for '{defense_name}': {exc}") from exc


class DefenseGridConfig(BaseModel):
    """Validated representation of a defense grid config entry with meta-fields.

    Extracts the meta-fields (evaluations, attacks, model config) from a
    grid config entry, separating them from the defense-specific hyperparameters
    that are passed to ``AlignmentDefenseConfig``.

    Attributes:
        defense_evals: Evaluations to run on the defended checkpoint.
        post_attack_evals: Evaluations to run on post-attack checkpoints.
        inference_model_config: Raw model configuration dict for inference.
            Populated from the ``model_config`` key in YAML.
        attacks: Attack specifications to stress-test the defense.

    Example:
        >>> grid_entry = {
        ...     "defense_evals": ["strong_reject"],
        ...     "post_attack_evals": ["strong_reject", "mmlu_pro_val"],
        ...     "model_config": {"template": "plain"},
        ...     "attacks": [{"name": "lora_finetune"}],
        ...     "learning_rate": 0.001,
        ... }
        >>> grid_config = DefenseGridConfig.from_dict(grid_entry)
        >>> grid_config.defense_evals
        [<EvalName.STRONG_REJECT: 'strong_reject'>]
    """

    model_config: ConfigDict = ConfigDict(  # pyright: ignore[reportIncompatibleVariableOverride]
        populate_by_name=True,
    )

    defense_evals: list[EvalName] = Field(..., description="Evaluations to run on the defended checkpoint")
    post_attack_evals: list[EvalName] = Field(..., description="Evaluations to run on post-attack checkpoints")
    inference_model_config: dict[str, object] = Field(
        ..., alias="model_config", description="Raw model configuration dict for inference"
    )
    attacks: list[PostDefenseAttackSpec] = Field(..., description="Attack specifications to stress-test the defense")

    @model_validator(mode="after")
    def _ensure_defense_evals(self) -> Self:
        """Ensure at least one defense evaluation is provided."""
        if not self.defense_evals:
            raise ValueError("Defense grid config must include at least one defense_eval entry.")
        return self

    @model_validator(mode="after")
    def _ensure_post_attack_evals(self) -> Self:
        """Ensure at least one post-attack evaluation is provided."""
        if not self.post_attack_evals:
            raise ValueError("Defense grid config must include at least one post_attack_eval entry.")
        return self

    @model_validator(mode="after")
    def _ensure_attacks(self) -> Self:
        """Ensure at least one attack specification is provided."""
        if not self.attacks:
            raise ValueError("Defense grid config must include at least one attack entry.")
        return self

    @classmethod
    def from_dict(cls, config_dict: dict[str, object]) -> Self:
        """Extract and validate meta-fields from a grid config entry.

        Picks out the meta-fields (``defense_evals``, ``post_attack_evals``,
        ``model_config``, ``attacks``) from a raw grid config dictionary.
        The remaining keys are defense-specific hyperparameters that should
        be passed to ``AlignmentDefenseConfig.from_dict()``.

        Args:
            config_dict: Raw config dictionary from a grid.yaml entry.

        Returns:
            Validated ``DefenseGridConfig`` containing only the meta-fields.

        Raises:
            ValueError: If required meta-fields are missing or invalid.

        Example:
            >>> entry = {
            ...     "defense_evals": ["strong_reject"],
            ...     "post_attack_evals": ["strong_reject"],
            ...     "model_config": {"template": "plain"},
            ...     "attacks": [{"name": "lora_finetune"}],
            ...     "alpha": 0.5,
            ... }
            >>> meta = DefenseGridConfig.from_dict(entry)
            >>> meta.attacks[0].name
            <AttackName.LORA_FINETUNE: 'lora_finetune'>
        """
        meta_keys = {
            str(ConfigKeys.DEFENSE_EVALS),
            str(ConfigKeys.POST_ATTACK_EVALS),
            str(ConfigKeys.MODEL_CONFIG),
            str(ConfigKeys.ATTACKS),
        }
        meta_payload = {k: v for k, v in config_dict.items() if k in meta_keys}

        try:
            return cls.model_validate(meta_payload)
        except ValidationError as exc:  # pragma: no cover - bubble detailed error
            raise ValueError(f"Invalid defense grid config meta-fields: {exc}") from exc


def load_defense_base_config(config_root: Path, defense_name: DefenseName) -> tuple[str, dict[str, object]]:
    """Load the default grid config for a defense.

    Loads the ``grid.yaml`` file for the specified defense and extracts the
    base configuration. Prefers a config named ``"base"`` if present;
    otherwise uses the first config in the file.

    Args:
        config_root: Root directory containing ``<defense>/grid.yaml`` files.
        defense_name: Name of the defense whose base config should be loaded.

    Returns:
        A tuple of ``(config_name, config_dict)`` where config_name is the key
        used (``"base"`` if available) and config_dict is the full configuration
        payload including both meta-fields and defense hyperparameters.

    Raises:
        TypeError: If the config value is not a dictionary.

    Example:
        >>> config_root = Path("configs/whitebox/defenses")
        >>> name, config = load_defense_base_config(config_root, DefenseName.CRL)
        >>> name
        'base'
        >>> "defense_evals" in config
        True
    """
    grid_path = Path(config_root, defense_name, ConfigPath.GRID_YAML)
    grid_payload = yaml_to_dict(grid_path)

    if ConfigKeys.BASE in grid_payload:
        base_cfg = grid_payload[ConfigKeys.BASE]

        if isinstance(base_cfg, dict):
            cfg_typed: dict[str, object] = base_cfg
            return str(ConfigKeys.BASE), cfg_typed

        raise TypeError(f"Expected dict for '{ConfigKeys.BASE}' config, got {type(base_cfg)}")

    first_key = next(iter(grid_payload.keys()))
    first_cfg = grid_payload[first_key]

    if isinstance(first_cfg, dict):
        cfg_typed = first_cfg
        return str(first_key), cfg_typed

    raise TypeError(f"Expected dict for config '{first_key}', got {type(first_cfg)}")
