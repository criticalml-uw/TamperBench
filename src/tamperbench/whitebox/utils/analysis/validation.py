"""Validation utilities for results directory structure, attack names, and required files."""

# pyright: reportExplicitAny=false

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from pathlib import Path

import optuna

from tamperbench.whitebox.attacks.registry import ATTACKS_REGISTRY
from tamperbench.whitebox.utils.analysis.config import AnalysisConfig
from tamperbench.whitebox.utils.benchmark.path_generation import (
    get_single_objective_study_name,
)
from tamperbench.whitebox.utils.names import (
    AttackName,
    BestJsonKeys,
    OptunaPaths,
    TrialSummaryKeys,
)

LOGGER = logging.getLogger(__name__)


class IgnoredDirPrefix(str, Enum):
    """Directory prefixes to ignore during validation."""

    AGGREGATED = "aggregated"
    EPSILON = "epsilon_"
    CONFIGS = "configs_"


@dataclass
class ValidationResult:
    """Result of directory validation."""

    valid: bool
    errors: list[str] = dataclass_field(default_factory=list)
    warnings: list[str] = dataclass_field(default_factory=list)
    models_found: list[str] = dataclass_field(default_factory=list)
    attacks_found: dict[str, list[AttackName]] = dataclass_field(default_factory=dict)
    skipped_models: list[str] = dataclass_field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.valid = False
        LOGGER.error(message)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        LOGGER.warning(message)


class ResultsValidator:
    """Validate results directory structure and contents."""

    config: AnalysisConfig

    def __init__(self, config: AnalysisConfig):
        """Initialize validator with analysis configuration.

        Args:
            config: Analysis configuration with model inclusion/exclusion settings.

        Example:
            >>> config = AnalysisConfig.from_yaml(Path("config.yaml"))
            >>> validator = ResultsValidator(config)
        """
        self.config = config

    def validate_directory(
        self,
        results_dir: Path,
        require_study_db: bool = True,
        require_best_json: bool = False,
    ) -> ValidationResult:
        """Validate entire results directory structure and return discovered models/attacks.

        Checks directory existence, discovers model directories, validates each model's structure,
        and ensures at least one valid model exists. Raises ValueError if validation fails.

        Args:
            results_dir: Parent results directory to validate.
            require_study_db: If True, require study.db for each attack.
            require_best_json: If True, require best.json for each attack.

        Returns:
            ValidationResult with discovered models and attacks.

        Raises:
            ValueError: If directory validation fails.

        Example:
            >>> validator = ResultsValidator(config)
            >>> result = validator.validate_directory(Path("results/sweep_001"))
            >>> print(result.models_found)
            ['llama-7b', 'mistral-7b']
        """
        result = ValidationResult(valid=True)

        if not self._check_directory_exists(results_dir, result):
            return result

        model_dirs = self._find_model_directories(results_dir)
        if not model_dirs:
            result.add_warning(f"No model directories found in {results_dir}")
            return result

        self._validate_all_models(model_dirs, result, require_study_db, require_best_json)
        self._ensure_models_found(result)
        self._raise_if_invalid(result)

        LOGGER.info(f"Validation passed: {len(result.models_found)} models found")
        return result

    def validate_attack_name(self, attack_name: str) -> AttackName:
        """Validate attack name string against the ATTACKS_REGISTRY.

        Args:
            attack_name: Attack name string to validate.

        Returns:
            Validated AttackName enum value.

        Raises:
            ValueError: If attack name is invalid or not in registry.

        Example:
            >>> validator.validate_attack_name("lora_finetune")
            AttackName.LORA_FINETUNE
        """
        attack = self._parse_attack_name(attack_name)
        self._check_attack_in_registry(attack, attack_name)
        return attack

    def check_study_db(self, study_path: Path, _model_name: str, attack_name: AttackName) -> None:
        """Check that study.db exists and can be loaded with valid trials.

        Args:
            study_path: Path to study.db file.
            _model_name: Model name (unused, kept for API consistency).
            attack_name: Attack name for study name lookup.

        Raises:
            FileNotFoundError: If study.db doesn't exist.
            ValueError: If study.db is invalid or has no trials.

        Example:
            >>> validator.check_study_db(Path("results/model/attack/optuna_single/study.db"), "model", AttackName.LORA_FINETUNE)
        """
        if not study_path.exists():
            raise FileNotFoundError(f"Missing study.db at {study_path}")
        self._verify_study_integrity(study_path, attack_name)

    def check_best_json(self, best_json_path: Path, _model_name: str, _attack_name: AttackName) -> None:
        """Check that best.json exists and has valid structure with required fields.

        Args:
            best_json_path: Path to best.json file.
            _model_name: Model name (unused, kept for API consistency).
            _attack_name: Attack name (unused, kept for API consistency).

        Raises:
            FileNotFoundError: If best.json doesn't exist.
            ValueError: If best.json format is invalid.

        Example:
            >>> validator.check_best_json(Path("results/model/attack/optuna_single/best.json"), "model", AttackName.LORA_FINETUNE)
        """
        if not best_json_path.exists():
            raise FileNotFoundError(f"Missing best.json at {best_json_path}")
        self._validate_best_json_structure(best_json_path)

    def _check_directory_exists(self, results_dir: Path, result: ValidationResult) -> bool:
        """Return True if results_dir exists and is a directory, else add error and return False."""
        if not results_dir.exists():
            result.add_error(f"Results directory does not exist: {results_dir}")
            self._raise_if_invalid(result)
            return False

        if not results_dir.is_dir():
            result.add_error(f"Results path is not a directory: {results_dir}")
            self._raise_if_invalid(result)
            return False

        return True

    def _find_model_directories(self, results_dir: Path) -> list[Path]:
        """Return model directories in results_dir, excluding ignored prefixes (aggregated, epsilon_, configs_)."""
        return [d for d in results_dir.iterdir() if d.is_dir() and not self._is_ignored_directory(d.name)]

    def _is_ignored_directory(self, dir_name: str) -> bool:
        """Return True if dir_name starts with any IgnoredDirPrefix value."""
        return any(dir_name.startswith(prefix.value) for prefix in IgnoredDirPrefix)

    def _validate_all_models(
        self,
        model_dirs: list[Path],
        result: ValidationResult,
        require_study_db: bool,
        require_best_json: bool,
    ) -> None:
        """Validate each model directory and merge results into the ValidationResult."""
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name

            if not self.config.models.should_include_model(model_name):
                result.skipped_models.append(model_name)
                LOGGER.debug(f"Skipping model {model_name} (excluded by configuration)")
                continue

            self._validate_and_merge_model(model_dir, model_name, result, require_study_db, require_best_json)

    def _validate_and_merge_model(
        self,
        model_dir: Path,
        model_name: str,
        result: ValidationResult,
        require_study_db: bool,
        require_best_json: bool,
    ) -> None:
        """Validate a single model directory and merge its results into the parent ValidationResult."""
        model_result = self._validate_model_directory(model_dir, model_name, require_study_db, require_best_json)

        result.errors.extend(model_result.errors)
        result.warnings.extend(model_result.warnings)
        result.models_found.append(model_name)
        result.attacks_found[model_name] = model_result.attacks_found.get(model_name, [])

        if not model_result.valid:
            result.valid = False

    def _ensure_models_found(self, result: ValidationResult) -> None:
        """Add error to result if no models were found after filtering."""
        if not result.models_found:
            result.add_error("No valid model directories found after applying inclusion/exclusion filters")

    def _raise_if_invalid(self, result: ValidationResult) -> None:
        """Raise ValueError with logged errors if result.valid is False."""
        if not result.valid:
            LOGGER.error(f"Validation failed with {len(result.errors)} errors")
            for error in result.errors:
                LOGGER.error(f"  - {error}")
            raise ValueError("Directory validation failed")

    def _validate_model_directory(
        self,
        model_dir: Path,
        model_name: str,
        require_study_db: bool,
        require_best_json: bool,
    ) -> ValidationResult:
        """Validate a single model directory and return a ValidationResult with discovered attacks."""
        result = ValidationResult(valid=True)
        result.models_found = [model_name]
        result.attacks_found[model_name] = []

        attack_dirs = self._find_attack_directories(model_dir)
        if not attack_dirs:
            result.add_warning(f"No attack directories found for model {model_name}")
            return result

        self._validate_all_attacks(attack_dirs, model_name, result, require_study_db, require_best_json)

        return result

    def _find_attack_directories(self, model_dir: Path) -> list[Path]:
        """Return attack directories in model_dir, excluding ignored prefixes."""
        return [d for d in model_dir.iterdir() if d.is_dir() and not self._is_ignored_directory(d.name)]

    def _validate_all_attacks(
        self,
        attack_dirs: list[Path],
        model_name: str,
        result: ValidationResult,
        require_study_db: bool,
        require_best_json: bool,
    ) -> None:
        """Validate each attack directory and add valid attacks to result.attacks_found."""
        for attack_dir in sorted(attack_dirs):
            attack_name_str = attack_dir.name

            attack_name = self._try_validate_attack_name(attack_name_str, model_name, result)
            if attack_name is None:
                continue

            if not self._validate_attack_structure(
                attack_dir,
                model_name,
                attack_name,
                result,
                require_study_db,
                require_best_json,
            ):
                continue

            result.attacks_found[model_name].append(attack_name)

    def _try_validate_attack_name(
        self, attack_name_str: str, model_name: str, result: ValidationResult
    ) -> AttackName | None:
        """Return validated AttackName or None if invalid, adding error to result on failure."""
        try:
            return self.validate_attack_name(attack_name_str)
        except ValueError as e:
            result.add_error(f"{model_name}/{attack_name_str}: {e}")
            return None

    def _validate_attack_structure(
        self,
        attack_dir: Path,
        model_name: str,
        attack_name: AttackName,
        result: ValidationResult,
        require_study_db: bool,
        require_best_json: bool,
    ) -> bool:
        """Return True if attack directory has valid structure and required files, else add warning and return False."""
        optuna_single_dir = attack_dir / OptunaPaths.OPTUNA_SINGLE_DIR
        if not optuna_single_dir.exists():
            result.add_warning(f"{model_name}/{attack_name}: Missing {OptunaPaths.OPTUNA_SINGLE_DIR} directory")
            return False

        if not self._check_required_files(
            optuna_single_dir,
            model_name,
            attack_name,
            result,
            require_study_db,
            require_best_json,
        ):
            return False

        return True

    def _check_required_files(
        self,
        optuna_single_dir: Path,
        model_name: str,
        attack_name: AttackName,
        result: ValidationResult,
        require_study_db: bool,
        require_best_json: bool,
    ) -> bool:
        """Return True if required files (study.db, best.json) exist and are valid, else add error and return False."""
        if require_study_db:
            study_db_path = optuna_single_dir / OptunaPaths.STUDY_DB
            if not self._try_check_study_db(study_db_path, model_name, attack_name, result):
                return False

        if require_best_json:
            best_json_path = optuna_single_dir / OptunaPaths.BEST_JSON
            if not self._try_check_best_json(best_json_path, model_name, attack_name, result):
                return False

        return True

    def _try_check_study_db(
        self,
        study_db_path: Path,
        model_name: str,
        attack_name: AttackName,
        result: ValidationResult,
    ) -> bool:
        """Return True if study.db check passes, else add error to result and return False."""
        try:
            self.check_study_db(study_db_path, model_name, attack_name)
            return True
        except (FileNotFoundError, ValueError) as e:
            result.add_error(f"{model_name}/{attack_name}: {e}")
            return False

    def _try_check_best_json(
        self,
        best_json_path: Path,
        model_name: str,
        attack_name: AttackName,
        result: ValidationResult,
    ) -> bool:
        """Return True if best.json check passes, else add error to result and return False."""
        try:
            self.check_best_json(best_json_path, model_name, attack_name)
            return True
        except (FileNotFoundError, ValueError) as e:
            result.add_error(f"{model_name}/{attack_name}: {e}")
            return False

    def _parse_attack_name(self, attack_name: str) -> AttackName:
        """Parse attack_name string to AttackName enum, raising ValueError if invalid."""
        try:
            return AttackName(attack_name)
        except ValueError as e:
            available = [a.value for a in AttackName]  # pyright: ignore[reportAny]
            raise ValueError(f"Invalid attack name: '{attack_name}'. Available attacks: {available}") from e

    def _check_attack_in_registry(self, attack: AttackName, attack_name: str) -> None:
        """Raise ValueError if attack is not registered in ATTACKS_REGISTRY."""
        if attack not in ATTACKS_REGISTRY:
            available = list(ATTACKS_REGISTRY.keys())
            msg = f"Attack '{attack_name}' not found in ATTACKS_REGISTRY. Available attacks: {available}"
            raise ValueError(msg)

    def _verify_study_integrity(self, study_path: Path, attack_name: AttackName) -> None:
        """Load study from study_path and raise ValueError if it fails or has no trials."""
        try:
            study_name = self._get_study_name(study_path, attack_name)
            storage_url = f"sqlite:///{study_path}"
            study = optuna.load_study(study_name=study_name, storage=storage_url)

            if len(study.trials) == 0:
                raise ValueError(f"Study {study_name} has no trials")

        except Exception as e:
            raise ValueError(f"Failed to load study from {study_path}: {e.__class__.__name__}: {e}") from e

    def _get_study_name(self, study_path: Path, attack_name: AttackName) -> str:
        """Return study name from best.json if present, otherwise generate default from attack_name."""
        best_json_path = study_path.parent / OptunaPaths.BEST_JSON
        if best_json_path.exists():
            with best_json_path.open() as f:
                best_data: dict[str, object] = json.load(f)  # pyright: ignore[reportAny]
                study_name = best_data.get(BestJsonKeys.STUDY_NAME)
                if isinstance(study_name, str):
                    return study_name
                return get_single_objective_study_name(attack_name)
        return get_single_objective_study_name(attack_name)

    def _validate_best_json_structure(self, best_json_path: Path) -> None:
        """Load and validate best.json structure, raising ValueError if invalid."""
        try:
            with best_json_path.open() as f:
                best_data: dict[str, object] = json.load(f)  # pyright: ignore[reportAny]

            self._check_required_fields(best_data)
            self._validate_top_trials(best_data)

        except json.JSONDecodeError as e:
            raise ValueError(f"best.json is not valid JSON: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to validate best.json: {e}") from e

    def _check_required_fields(self, best_data: dict[str, object]) -> None:
        """Raise ValueError if best_data is missing any required BestJsonKeys fields."""
        required_fields = [
            BestJsonKeys.STUDY_NAME,
            BestJsonKeys.BASE_CONFIG,
            BestJsonKeys.DIRECTIONS,
            BestJsonKeys.TOP_TRIALS,
        ]
        for required_field in required_fields:
            if required_field not in best_data:
                raise ValueError(f"best.json missing required field: {required_field}")

    def _validate_top_trials(self, best_data: dict[str, object]) -> None:
        """Raise ValueError if top_trials is not a non-empty list with valid trial structure."""
        top_trials = best_data[BestJsonKeys.TOP_TRIALS]
        if not isinstance(top_trials, list):
            raise ValueError(f"best.json: {BestJsonKeys.TOP_TRIALS} must be a list")

        if len(top_trials) == 0:
            raise ValueError(f"best.json: {BestJsonKeys.TOP_TRIALS} is empty")

        first_trial = top_trials[0]
        if isinstance(first_trial, dict):
            self._validate_trial_fields(first_trial)

    def _validate_trial_fields(self, trial: dict[str, object]) -> None:
        """Raise ValueError if trial is missing required TrialSummaryKeys fields."""
        required_trial_fields = [
            TrialSummaryKeys.RANK,
            TrialSummaryKeys.TRIAL_NUMBER,
            TrialSummaryKeys.PARAMS,
        ]
        for required_field in required_trial_fields:
            if required_field not in trial:
                msg = f"best.json: {BestJsonKeys.TOP_TRIALS}[0] missing required field: {required_field}"
                raise ValueError(msg)
