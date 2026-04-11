"""Heatmap data structure builders for aggregating and organizing sweep results.

Provides HeatmapBuilder for constructing model-by-attack matrices with raw values
and deltas from baseline, plus HeatmapData for serialization.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tamperbench.whitebox.utils.analysis.collectors import StudyData
from tamperbench.whitebox.utils.analysis.config import AnalysisConfig
from tamperbench.whitebox.utils.names import (
    AttackName,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class HeatmapData:
    """Heatmap data structure with raw and delta matrices.

    Attributes:
        safety_raw: Raw safety metric values (models x attacks)
        safety_delta: Safety deltas from baseline (models x attacks)
        utility_raw: Raw utility metric values (models x attacks)
        utility_delta: Utility deltas from baseline (models x attacks)
        models: Ordered list of model names
        attacks: Ordered list of attack names
        metadata: Optional metadata dictionary
    """

    safety_raw: NDArray[np.float32]
    safety_delta: NDArray[np.float32]
    utility_raw: NDArray[np.float32]
    utility_delta: NDArray[np.float32]
    models: list[str]
    attacks: list[AttackName]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Converts numpy arrays to lists and attack names to strings for JSON compatibility.

        Returns:
            JSON-serializable dictionary representation

        Example:
            >>> data = HeatmapData(safety_raw=np.array([[0.5]]), ...)
            >>> json_data = data.to_dict()
            >>> json.dumps(json_data)  # Now JSON-serializable
        """
        return {
            "safety_raw": self._to_list(self.safety_raw),
            "safety_delta": self._to_list(self.safety_delta),
            "utility_raw": self._to_list(self.utility_raw),
            "utility_delta": self._to_list(self.utility_delta),
            "models": self.models,
            "attacks": [str(a) for a in self.attacks],
            "metadata": self.metadata,
        }

    def _to_list(self, array: NDArray[np.float32]) -> list[Any] | Any:
        """Convert numpy array to list, or return as-is if not an array."""
        return array.tolist() if hasattr(array, "tolist") else array


class HeatmapBuilder:
    """Build heatmap data structures from collected data."""

    def __init__(self, config: AnalysisConfig):
        """Initialize builder with configuration.

        Args:
            config: Analysis configuration
        """
        self.config: AnalysisConfig = config

    def order_models(self, model_names: list[str]) -> list[str]:
        """Order models according to preferred order from config.

        Models in preferred_order appear first, maintaining that order.
        Remaining models are sorted alphabetically after the preferred ones.

        Args:
            model_names: List of model names to order

        Returns:
            Sorted list with preferred models first, then alphabetical

        Example:
            >>> builder = HeatmapBuilder(config)
            >>> builder.order_models(["mistral-7b", "llama-8b", "qwen-7b"])
            ['qwen-7b', 'llama-8b', 'mistral-7b']  # Based on config.models.preferred_order
        """
        preferred = self.config.models.preferred_order
        return sorted(model_names, key=self._create_model_sort_key(preferred))

    def order_attacks(self, aggregated_data: dict[str, dict[AttackName, StudyData]]) -> list[AttackName]:
        """Order attacks according to preferred order from config.

        Attacks in preferred_order appear first, maintaining that order.
        Remaining attacks are sorted alphabetically after the preferred ones.

        Args:
            aggregated_data: Nested dict mapping model names to attack data

        Returns:
            Sorted list of AttackName with preferred attacks first

        Example:
            >>> builder = HeatmapBuilder(config)
            >>> all_data = {"model1": {AttackName.LORA_FINETUNE: study_data}}
            >>> builder.order_attacks(all_data)
            [AttackName.NO_WEIGHT_MODIFICATION, AttackName.LORA_FINETUNE, ...]
        """
        all_attacks = self._collect_unique_attacks(aggregated_data)
        preferred = self.config.attacks.preferred_order

        ordered = self._get_preferred_attacks(preferred, all_attacks)
        ordered.extend(self._get_remaining_attacks(ordered, all_attacks))

        return ordered

    def calculate_deltas(
        self,
        raw_matrix: NDArray[np.float32],
        attacks: list[AttackName],
        baseline_attack: AttackName,
    ) -> NDArray[np.float32]:
        """Calculate delta matrix (difference from baseline).

        Computes element-wise difference between each value and the corresponding
        baseline value. Returns zeros if baseline attack is not found.

        Args:
            raw_matrix: Raw values matrix (models x attacks)
            attacks: Ordered list of attacks
            baseline_attack: Baseline attack to subtract

        Returns:
            Delta matrix with same shape as raw_matrix

        Example:
            >>> raw = np.array([[0.8, 0.9], [0.7, 0.85]])  # 2 models, 2 attacks
            >>> attacks = [AttackName.NO_WEIGHT_MODIFICATION, AttackName.LORA_FINETUNE]
            >>> deltas = builder.calculate_deltas(raw, attacks, AttackName.NO_WEIGHT_MODIFICATION)
            >>> deltas[0, 1]  # Delta for model 0, attack 1
            0.1
        """
        baseline_idx = self._find_baseline_index(attacks, baseline_attack)
        if baseline_idx is None:
            LOGGER.warning(f"Baseline attack {baseline_attack} not found, returning zeros")
            return np.zeros_like(raw_matrix)

        baseline_col = raw_matrix[:, baseline_idx]
        return raw_matrix - baseline_col[:, np.newaxis]

    def _create_model_sort_key(self, preferred: list[str]) -> Callable[[str], tuple[int, str]]:
        """Create sort key function prioritizing preferred order, then alphabetical."""

        def sort_key(model: str) -> tuple[int, str]:
            try:
                return (preferred.index(model), model)
            except ValueError:
                return (len(preferred), model)

        return sort_key

    def _collect_unique_attacks(self, aggregated_data: dict[str, dict[AttackName, StudyData]]) -> set[AttackName]:
        """Collect all unique attack names from nested model->attack data."""
        all_attacks: set[AttackName] = set()
        for model_data in aggregated_data.values():
            all_attacks.update(model_data.keys())
        return all_attacks

    def _get_preferred_attacks(self, preferred: list[AttackName], all_attacks: set[AttackName]) -> list[AttackName]:
        """Filter preferred attacks to only those present in data."""
        return [attack for attack in preferred if attack in all_attacks]

    def _get_remaining_attacks(self, ordered: list[AttackName], all_attacks: set[AttackName]) -> list[AttackName]:
        """Get attacks not in ordered list, sorted alphabetically."""
        remaining = all_attacks - set(ordered)
        return sorted(remaining, key=lambda a: str(a))

    def _find_baseline_index(self, attacks: list[AttackName], baseline_attack: AttackName) -> int | None:
        """Find index of baseline attack in list, or None if not found."""
        try:
            return attacks.index(baseline_attack)
        except ValueError:
            return None
