"""Configuration dataclasses for the analysis module.

Provides typed configuration classes for epsilon-bounded analysis, heatmap
generation, and visualization settings. All configs support YAML serialization
and validation against attack/evaluation registries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from safetunebed.whitebox.attacks.registry import ATTACKS_REGISTRY
from safetunebed.whitebox.utils.names import AttackName, OptimizationDirection


@dataclass
class AttackCategory:
    """Attack category grouping."""

    display_name: str
    attacks: list[AttackName]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttackCategory:  # pyright: ignore[reportExplicitAny]
        """Create from dictionary with validation."""
        attacks = [AttackName(a) for a in data["attacks"]]  # pyright: ignore[reportAny]
        return cls(display_name=data["display_name"], attacks=attacks)  # pyright: ignore[reportAny]


@dataclass
class AttackConfig:
    """Attack ordering and categorization configuration."""

    preferred_order: list[AttackName]
    categories: dict[str, AttackCategory]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttackConfig:  # pyright: ignore[reportExplicitAny]
        """Create from dictionary with validation.

        Validates that all attack names exist in the registry and parses attack categories.

        Args:
            data: Dictionary containing preferred_order and categories

        Returns:
            Validated AttackConfig instance

        Raises:
            ValueError: If an attack is not found in the registry

        Example:
            >>> data = {
            ...     "preferred_order": ["no_weight_modification", "lora_finetune"],
            ...     "categories": {"harmful": {"display_name": "Harmful", "attacks": ["lora_finetune"]}}
            ... }
            >>> config = AttackConfig.from_dict(data)
            >>> config.preferred_order[0]
            AttackName.NO_WEIGHT_MODIFICATION
        """
        preferred_order = cls._validate_attacks(data["preferred_order"])  # pyright: ignore[reportAny]
        categories = cls._parse_categories(data["categories"])  # pyright: ignore[reportAny]
        return cls(preferred_order=preferred_order, categories=categories)

    @classmethod
    def _validate_attacks(cls, attack_strings: list[str]) -> list[AttackName]:
        """Parse and validate attack names against the registry."""
        preferred_order = []
        for attack_str in attack_strings:
            attack_name = AttackName(attack_str)
            if attack_name not in ATTACKS_REGISTRY:
                raise ValueError(
                    f"Attack '{attack_name}' not found in ATTACKS_REGISTRY. "
                    f"Available attacks: {list(ATTACKS_REGISTRY.keys())}"
                )
            preferred_order.append(attack_name)
        return preferred_order

    @classmethod
    def _parse_categories(
        cls,
        categories_data: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    ) -> dict[str, AttackCategory]:
        """Parse category dict into AttackCategory instances."""
        categories: dict[str, AttackCategory] = {}
        for cat_key, cat_data in categories_data.items():  # pyright: ignore[reportAny]
            categories[cat_key] = AttackCategory.from_dict(cat_data)  # pyright: ignore[reportAny]
        return categories


@dataclass
class ModelGroup:
    """Model group for visualization."""

    display_name: str
    models: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelGroup:  # pyright: ignore[reportExplicitAny]
        """Create from dictionary."""
        return cls(
            display_name=data["display_name"],  # pyright: ignore[reportAny]
            models=data["models"],  # pyright: ignore[reportAny]
        )


@dataclass
class ModelConfig:
    """Model ordering configuration.

    Models must be explicitly listed in include_models to be analyzed.
    This makes it clear which models are being processed.
    """

    preferred_order: list[str]
    include_models: list[str]  # Required: explicit model list
    groups: dict[str, ModelGroup] | None = None  # Optional: model groups for visual separation

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:  # pyright: ignore[reportExplicitAny]
        """Create from dictionary.

        Parses model configuration including required model list and optional groups.

        Args:
            data: Dictionary containing model configuration

        Returns:
            ModelConfig instance

        Raises:
            ValueError: If include_models is not specified
        """
        include_models = cls._validate_include_models(data)
        groups = cls._parse_model_groups(data)

        return cls(
            preferred_order=data["preferred_order"],  # pyright: ignore[reportAny]
            include_models=include_models,
            groups=groups,
        )

    @classmethod
    def _validate_include_models(cls, data: dict[str, Any]) -> list[str]:  # pyright: ignore[reportExplicitAny]
        """Extract and validate the required include_models list."""
        include_models: list[str] | None = data.get("include_models")
        if not include_models:
            raise ValueError(
                "include_models must be specified in model configuration. Please explicitly list all models to analyze."
            )
        return include_models

    @classmethod
    def _parse_model_groups(cls, data: dict[str, Any]) -> dict[str, ModelGroup] | None:  # pyright: ignore[reportExplicitAny]
        """Parse optional model group configurations into ModelGroup instances."""
        if "groups" not in data:
            return None

        groups: dict[str, ModelGroup] = {}
        for group_key, group_data in data["groups"].items():  # pyright: ignore[reportAny]
            groups[group_key] = ModelGroup.from_dict(group_data)  # pyright: ignore[reportAny]
        return groups

    def should_include_model(self, model_name: str) -> bool:
        """Check if model should be included in analysis."""
        return model_name in self.include_models

    def get_model_group_boundaries(self) -> list[tuple[str, int, int]]:
        """Get model group boundaries for visualization.

        For each group, finds models that appear in preferred_order and determines
        their start and end indices for visual separation.

        Returns:
            List of (group_name, start_idx, end_idx) tuples based on preferred_order
        """
        if not self.groups:
            return []

        boundaries = []
        for _, group in self.groups.items():
            boundary = self._compute_group_boundary(group)
            if boundary is not None:
                boundaries.append(boundary)

        return boundaries

    def _compute_group_boundary(self, group: ModelGroup) -> tuple[str, int, int] | None:
        """Compute (display_name, start_idx, end_idx) for a model group."""
        group_models_in_order = [m for m in self.preferred_order if m in group.models]

        if not group_models_in_order:
            return None

        start_idx = self.preferred_order.index(group_models_in_order[0])
        end_idx = self.preferred_order.index(group_models_in_order[-1])
        return (group.display_name, start_idx, end_idx)


@dataclass
class MetricConfig:
    """Metric configuration with aliases and direction."""

    display_name: str
    aliases: list[str]
    direction: OptimizationDirection

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricConfig:  # pyright: ignore[reportExplicitAny]
        """Create from dictionary."""
        direction_str: str = data["direction"]  # pyright: ignore[reportAny]
        direction = OptimizationDirection.MAXIMIZE if direction_str == "maximize" else OptimizationDirection.MINIMIZE
        return cls(
            display_name=data["display_name"],  # pyright: ignore[reportAny]
            aliases=data["aliases"],  # pyright: ignore[reportAny]
            direction=direction,
        )


@dataclass
class FigureConfig:
    """Figure dimension configuration."""

    dpi: int
    width_formula: str
    height_formula: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FigureConfig:  # pyright: ignore[reportExplicitAny]
        """Create from dictionary."""
        return cls(
            dpi=data["dpi"],  # pyright: ignore[reportAny]
            width_formula=data["width_formula"],  # pyright: ignore[reportAny]
            height_formula=data["height_formula"],  # pyright: ignore[reportAny]
        )

    def calculate_width(self, n_columns: int) -> float:
        """Calculate figure width from formula.

        Evaluates width_formula with n_columns and max in scope.
        Uses simple eval - could be enhanced with safer expression parser if needed.

        Args:
            n_columns: Number of columns in the figure

        Returns:
            Calculated width in inches
        """
        result: float = eval(self.width_formula, {"n_columns": n_columns, "max": max})  # pyright: ignore[reportAny]
        return result

    def calculate_height(self, n_models: int) -> float:
        """Calculate figure height from formula.

        Evaluates height_formula with n_models and max in scope.

        Args:
            n_models: Number of models in the figure

        Returns:
            Calculated height in inches
        """
        result: float = eval(self.height_formula, {"n_models": n_models, "max": max})  # pyright: ignore[reportAny]
        return result


@dataclass
class VisualizationConfig:
    """Visualization settings for plots."""

    colormaps: dict[str, str]
    colormap_range: tuple[float, float]
    bad_color: tuple[float, float, float, float]
    figure: FigureConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualizationConfig:  # pyright: ignore[reportExplicitAny]
        """Create from dictionary."""
        colormap_range: tuple[float, float] = tuple(data["colormap_range"])  # pyright: ignore[reportAny]
        bad_color: tuple[float, float, float, float] = tuple(data["bad_color"])  # pyright: ignore[reportAny]
        return cls(
            colormaps=data["colormaps"],  # pyright: ignore[reportAny]
            colormap_range=colormap_range,
            bad_color=bad_color,
            figure=FigureConfig.from_dict(data["figure"]),  # pyright: ignore[reportAny]
        )


@dataclass
class HeatmapMethod:
    """Heatmap selection method configuration."""

    display_name: str
    priority: list[str] | None = None
    formula: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeatmapMethod:  # pyright: ignore[reportExplicitAny]
        """Create from dictionary."""
        return cls(
            display_name=data["display_name"],  # pyright: ignore[reportAny]
            priority=data.get("priority"),
            formula=data.get("formula"),
        )


@dataclass
class EpsilonBoundedConfig:
    """Epsilon-bounded analysis configuration."""

    baseline_attack: AttackName
    utility_metric: str  # Metric key to epsilon-bound (e.g., "mmlu_pro_val")
    safety_metric: str  # Metric key for attack success (e.g., "strong_reject")
    default_epsilons: list[float]
    default_n_trials: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpsilonBoundedConfig:  # pyright: ignore[reportExplicitAny]
        """Create from dictionary."""
        baseline_attack = AttackName(data["baseline_attack"])
        if baseline_attack not in ATTACKS_REGISTRY:
            raise ValueError(f"Baseline attack '{baseline_attack}' not found in ATTACKS_REGISTRY")
        return cls(
            baseline_attack=baseline_attack,
            utility_metric=data["utility_metric"],  # pyright: ignore[reportAny]
            safety_metric=data["safety_metric"],  # pyright: ignore[reportAny]
            default_epsilons=data["default_epsilons"],  # pyright: ignore[reportAny]
            default_n_trials=data["default_n_trials"],  # pyright: ignore[reportAny]
        )


@dataclass
class AnalysisConfig:
    """Main analysis configuration loaded from YAML.

    This is the root configuration object that contains all settings for the analysis
    pipeline. It is constructed from a YAML file and validates all settings against
    the attack and evaluation registries.
    """

    attacks: AttackConfig
    models: ModelConfig
    metrics: dict[str, MetricConfig]
    visualization: VisualizationConfig
    epsilon_bounded: EpsilonBoundedConfig
    heatmap_methods: dict[str, HeatmapMethod]

    @classmethod
    def from_yaml(cls, path: Path) -> AnalysisConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated AnalysisConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config contains invalid attack/eval names or settings

        Example:
            >>> config = AnalysisConfig.from_yaml(Path("configs/analysis/default.yaml"))
            >>> config.epsilon_bounded.default_epsilons
            [0.01, 0.03, 0.05]
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open() as f:
            data: dict[str, object] = yaml.safe_load(f)  # pyright: ignore[reportAny]

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisConfig:  # pyright: ignore[reportExplicitAny]
        """Create from dictionary with full validation.

        Parses all nested configuration sections including attacks, models, metrics,
        visualization settings, epsilon-bounded config, and heatmap methods.

        Args:
            data: Dictionary loaded from YAML

        Returns:
            Validated AnalysisConfig instance
        """
        attacks = AttackConfig.from_dict(data["attacks"])  # pyright: ignore[reportAny]
        models = ModelConfig.from_dict(data["models"])  # pyright: ignore[reportAny]
        metrics = cls._parse_metrics(data["metrics"])  # pyright: ignore[reportAny]
        visualization = VisualizationConfig.from_dict(data["visualization"])  # pyright: ignore[reportAny]
        epsilon_bounded = EpsilonBoundedConfig.from_dict(data["epsilon_bounded"])  # pyright: ignore[reportAny]
        heatmap_methods = cls._parse_heatmap_methods(data["heatmap_methods"])  # pyright: ignore[reportAny]

        return cls(
            attacks=attacks,
            models=models,
            metrics=metrics,
            visualization=visualization,
            epsilon_bounded=epsilon_bounded,
            heatmap_methods=heatmap_methods,
        )

    @classmethod
    def _parse_metrics(cls, metrics_data: dict[str, Any]) -> dict[str, MetricConfig]:  # pyright: ignore[reportExplicitAny]
        """Parse metric dict into MetricConfig instances."""
        metrics: dict[str, MetricConfig] = {}
        for metric_key, metric_data in metrics_data.items():  # pyright: ignore[reportAny]
            metrics[metric_key] = MetricConfig.from_dict(metric_data)  # pyright: ignore[reportAny]
        return metrics

    @classmethod
    def _parse_heatmap_methods(
        cls,
        methods_data: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    ) -> dict[str, HeatmapMethod]:
        """Parse heatmap method dict into HeatmapMethod instances."""
        heatmap_methods: dict[str, HeatmapMethod] = {}
        for method_key, method_data in methods_data.items():  # pyright: ignore[reportAny]
            heatmap_methods[method_key] = HeatmapMethod.from_dict(method_data)  # pyright: ignore[reportAny]
        return heatmap_methods

    def get_metric_by_alias(self, alias: str) -> MetricConfig | None:
        """Find metric config by alias (e.g., 'strong_reject' -> MetricConfig)."""
        for metric in self.metrics.values():
            if alias in metric.aliases:
                return metric
        return None
