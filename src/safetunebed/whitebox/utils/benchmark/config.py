"""Configuration models and loaders for Optuna sweeps."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator
from typing_extensions import Self

from safetunebed.whitebox.evals.registry import EVALS_REGISTRY
from safetunebed.whitebox.utils import AttackName, ConfigPath, EvalName
from safetunebed.whitebox.utils.benchmark.io import yaml_to_dict
from safetunebed.whitebox.utils.names import ConfigKeys


class AttackSweepConfig(BaseModel):
    """Validated representation of a single attack's Optuna sweep file."""

    evals: list[EvalName] = Field(..., description="Ordered list of eval names to optimize against")
    sweep: dict[str, object] = Field(..., description="Nested Optuna search space definition")

    @model_validator(mode="after")
    def _ensure_evals(self) -> Self:
        """Ensure at least one evaluation is provided."""
        if not self.evals:
            raise ValueError("Sweep config must include at least one eval entry.")

        return self

    @property
    def primary_objective_direction(self) -> str:
        """Get optimization direction from the primary (first) evaluation.

        Returns:
            Optimization direction string, either "maximize" or "minimize".

        Example:
            >>> sweep = AttackSweepConfig.from_yaml(config_root, AttackName.LORA_FINETUNE)
            >>> sweep.primary_objective_direction
            'maximize'
        """
        return EVALS_REGISTRY[self.evals[0]].attacker_direction.value  # pyright: ignore[reportAny]

    @classmethod
    def from_yaml(
        cls,
        config_root: Path,
        attack_name: AttackName,
        sweep_filename: str | ConfigPath = ConfigPath.SWEEP,
    ) -> Self:
        """Load and validate an Optuna sweep specification from YAML.

        Loads and validates a sweep configuration file (e.g., ``sweep.yaml`` or
        ``single_objective_sweep.yaml``) that defines the hyperparameter search
        space and target evaluations for Optuna optimization.

        Args:
            config_root: Root directory containing per-attack sweep config files.
            attack_name: Name of the attack whose sweep configuration should be loaded.
            sweep_filename: Name of the sweep config file (default: ``ConfigPath.SWEEP``).

        Returns:
            Validated ``AttackSweepConfig`` containing ``evals`` list and ``sweep`` dict.

        Raises:
            FileNotFoundError: If the sweep YAML file is missing.
            ValueError: If validation of the sweep payload fails.

        Example:
            >>> config_root = Path("configs/whitebox/attacks")
            >>> sweep = AttackSweepConfig.from_yaml(
            ...     config_root,
            ...     AttackName.LORA_FINETUNE,
            ...     ConfigPath.SINGLE_OBJECTIVE_SWEEP,
            ... )
            >>> sweep.evals
            [<EvalName.STRONG_REJECT: 'strong_reject'>]
            >>> "learning_rate" in sweep.sweep
            True
        """
        sweep_path = Path(config_root, attack_name, sweep_filename)

        if not sweep_path.exists():
            raise FileNotFoundError(f"Missing sweep config at: {sweep_path}")

        payload = yaml_to_dict(sweep_path)

        try:
            return cls.model_validate(payload)
        except ValidationError as exc:  # pragma: no cover - bubble detailed error
            raise ValueError(f"Invalid sweep config for '{attack_name}': {exc}") from exc


def load_attack_base_config(config_root: Path, attack_name: AttackName) -> tuple[str, dict[str, object]]:
    """Load the default grid config for an attack.

    Loads the ``grid.yaml`` file for the specified attack and extracts the base
    configuration. Prefers a config named ``"base"`` if present; otherwise uses
    the first config in the file.

    Args:
        config_root: Root directory containing ``<attack>/grid.yaml`` files.
        attack_name: Name of the attack whose base config should be loaded.

    Returns:
        A tuple of ``(config_name, config_dict)`` where config_name is the key
        used (``"base"`` if available) and config_dict is the configuration payload.

    Raises:
        TypeError: If the config value is not a dictionary.

    Example:
        >>> config_root = Path("configs/whitebox/attacks")
        >>> name, config = load_attack_base_config(config_root, AttackName.LORA_FINETUNE)
        >>> name
        'base'
        >>> config["learning_rate"]
        0.0001
    """
    grid_path = Path(config_root, attack_name, ConfigPath.GRID_YAML)
    grid_payload = yaml_to_dict(grid_path)

    if ConfigKeys.BASE in grid_payload:
        base_cfg = grid_payload[ConfigKeys.BASE]

        if isinstance(base_cfg, dict):
            cfg_typed: dict[str, object] = base_cfg  # pyright: ignore[reportUnknownVariableType]
            return str(ConfigKeys.BASE), cfg_typed

        raise TypeError(f"Expected dict for '{ConfigKeys.BASE}' config, got {type(base_cfg)}")

    first_key = next(iter(grid_payload.keys()))
    first_cfg = grid_payload[first_key]

    if isinstance(first_cfg, dict):
        cfg_typed = first_cfg  # pyright: ignore[reportUnknownVariableType]
        return str(first_key), cfg_typed

    raise TypeError(f"Expected dict for config '{first_key}', got {type(first_cfg)}")
