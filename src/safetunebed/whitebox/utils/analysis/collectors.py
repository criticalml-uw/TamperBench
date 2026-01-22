"""Data collection from Optuna study databases.

Provides OptunaCollector for loading studies and collecting data across multiple
models and attacks in a results directory hierarchy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import optuna

from safetunebed.whitebox.utils.analysis.config import AnalysisConfig
from safetunebed.whitebox.utils.analysis.io import find_models
from safetunebed.whitebox.utils.benchmark.path_generation import StudyPaths
from safetunebed.whitebox.utils.names import AttackName

LOGGER = logging.getLogger(__name__)


@dataclass
class StudyData:
    """Container for Optuna study data.

    Attributes:
        study: The loaded Optuna study object
    """

    study: optuna.Study


class OptunaCollector:
    """Collect data from Optuna study databases."""

    config: AnalysisConfig

    def __init__(self, config: AnalysisConfig):
        """Initialize collector with configuration.

        Args:
            config: Analysis configuration
        """
        self.config = config

    def load_study(
        self,
        results_dir: Path,
        model_name: str,
        attack_name: AttackName,
    ) -> optuna.Study:
        """Load Optuna study from database.

        Args:
            results_dir: Parent results directory.
            model_name: Model name (alias) directory.
            attack_name: Attack name.

        Returns:
            Loaded Optuna study.

        Raises:
            FileNotFoundError: If study directory or study.db doesn't exist.
            ValueError: If study cannot be loaded.

        Example:
            >>> collector = OptunaCollector(config)
            >>> study = collector.load_study(Path("results"), "llama-8b", AttackName.LORA_FINETUNE)
            >>> len(study.trials)
            50
        """
        study_paths = StudyPaths.from_existing(results_dir, model_name, attack_name)

        try:
            study = optuna.load_study(study_name=study_paths.study_name, storage=study_paths.storage_url)
        except Exception as e:
            raise ValueError(
                f"Failed to load study {study_paths.study_name} from {study_paths.storage_path}: {e}"
            ) from e

        LOGGER.debug(f"Loaded study {study_paths.study_name} with {len(study.trials)} trials")
        return study

    def collect_attack_data(
        self,
        results_dir: Path,
        model_name: str,
        attack_name: AttackName,
    ) -> StudyData:
        """Collect all data for a specific attack.

        Loads the Optuna study from the database and wraps it in a StudyData structure.

        Args:
            results_dir: Parent results directory.
            model_name: Model name (alias) directory.
            attack_name: Attack name.

        Returns:
            StudyData with all extracted information.
        """
        study = self.load_study(results_dir, model_name, attack_name)
        return StudyData(study=study)

    def collect_model_data(
        self,
        results_dir: Path,
        model_name: str,
        attack_names: list[AttackName] | None = None,
    ) -> dict[AttackName, StudyData]:
        """Collect data for all attacks in a model.

        If attack_names is not specified, uses the preferred attack order from config.
        Skips attacks that cannot be loaded (missing files or invalid data).

        Args:
            results_dir: Parent results directory.
            model_name: Model name (alias) directory.
            attack_names: Optional list of attacks to collect (None = all).

        Returns:
            Dictionary mapping attack names to StudyData.

        Example:
            >>> collector = OptunaCollector(config)
            >>> model_data = collector.collect_model_data(Path("results"), "llama-8b")
            >>> AttackName.LORA_FINETUNE in model_data
            True
        """
        attack_names = self._resolve_attack_names(attack_names)
        data: dict[AttackName, StudyData] = {}

        for attack_name in attack_names:
            study_data = self._try_collect_attack(results_dir, model_name, attack_name)
            if study_data is not None:
                data[attack_name] = study_data

        return data

    def collect_all_models(
        self,
        parent_results_dir: Path,
        model_names: list[str] | None = None,
        attack_names: list[AttackName] | None = None,
    ) -> dict[str, dict[AttackName, StudyData]]:
        """Collect data for all models in directory.

        If model_names is not specified, discovers all models in the directory
        and filters them according to the config's model inclusion rules.

        Args:
            parent_results_dir: Parent results directory.
            model_names: Optional list of models to collect (None = all in config).
            attack_names: Optional list of attacks to collect (None = all).

        Returns:
            Nested dictionary: {model_name: {attack_name: StudyData}}

        Example:
            >>> collector = OptunaCollector(config)
            >>> all_data = collector.collect_all_models(Path("results/sweep_001"))
            >>> list(all_data.keys())
            ['llama-8b', 'mistral-7b', 'qwen-7b']
        """
        LOGGER.info("Collecting data from all models...")

        model_names = self._resolve_model_names(parent_results_dir, model_names)
        all_data = self._collect_data_for_models(parent_results_dir, model_names, attack_names)

        LOGGER.info(f"Collected data for {len(all_data)} models")
        return all_data

    def _resolve_attack_names(self, attack_names: list[AttackName] | None) -> list[AttackName]:
        """Return attack_names if provided, otherwise use config's preferred order."""
        return attack_names if attack_names is not None else self.config.attacks.preferred_order

    def _try_collect_attack(
        self,
        results_dir: Path,
        model_name: str,
        attack_name: AttackName,
    ) -> StudyData | None:
        """Attempt to collect attack data, returning None on failure."""
        try:
            study_data = self.collect_attack_data(results_dir, model_name, attack_name)
            LOGGER.info(f"Collected data for {model_name}/{attack_name}")
            return study_data
        except (FileNotFoundError, ValueError) as e:
            LOGGER.warning(f"Skipping {model_name}/{attack_name}: {e}")
            return None

    def _resolve_model_names(self, parent_results_dir: Path, model_names: list[str] | None) -> list[str]:
        """Return model_names if provided, otherwise discover and filter by config."""
        if model_names is not None:
            return model_names

        all_models = find_models(parent_results_dir)
        return [m for m in all_models if self.config.models.should_include_model(m)]

    def _collect_data_for_models(
        self,
        parent_results_dir: Path,
        model_names: list[str],
        attack_names: list[AttackName] | None,
    ) -> dict[str, dict[AttackName, StudyData]]:
        """Iterate over model_names and collect attack data for each."""
        all_data: dict[str, dict[AttackName, StudyData]] = {}

        for model_name in model_names:
            LOGGER.info(f"Collecting data for model: {model_name}")
            model_data = self.collect_model_data(parent_results_dir, model_name, attack_names)
            if model_data:
                all_data[model_name] = model_data

        return all_data
