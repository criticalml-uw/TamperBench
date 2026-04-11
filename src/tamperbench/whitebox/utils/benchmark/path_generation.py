"""Path and naming helpers for Optuna trials and results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tamperbench.whitebox.utils import AttackName
from tamperbench.whitebox.utils.names import DefenseName, EvalPath, OptunaPaths


@dataclass
class StudyPaths:
    """Standardized paths for an Optuna sweep study.

    Encapsulates all path generation for Optuna sweeps, ensuring consistent
    directory structure across runs.

    Structure::

        results_dir/
        └── model_alias/
            └── attack_name/
                └── sweep_subdir/
                    ├── study.db
                    ├── best.yaml
                    ├── best.json
                    └── trial_0/
                        └── ...

    Attributes:
        results_dir: Root directory for all sweep results.
        model_alias: Identifier for the model being attacked.
        attack_name: Name of the attack being swept.
        sweep_subdir: Subdirectory for this sweep type (default: single-objective).

    Example:
        >>> paths = StudyPaths(
        ...     results_dir=Path("results"),
        ...     model_alias="llama-3-8b_2026_01_14",
        ...     attack_name=AttackName.LORA_FINETUNE,
        ... )
        >>> paths.sweep_results_dir
        PosixPath('results/llama-3-8b_2026_01_14/lora_finetune/optuna_single')
        >>> paths.storage_url
        'sqlite:///results/llama-3-8b_2026_01_14/lora_finetune/optuna_single/study.db'
    """

    results_dir: Path
    model_alias: str
    attack_name: AttackName
    sweep_subdir: str = OptunaPaths.OPTUNA_SINGLE_DIR

    @property
    def model_results_dir(self) -> Path:
        """Directory for all results related to this model."""
        return self.results_dir / self.model_alias

    @property
    def sweep_results_dir(self) -> Path:
        """Directory for this attack's sweep results."""
        return self.model_results_dir / str(self.attack_name) / self.sweep_subdir

    @property
    def storage_path(self) -> Path:
        """Path to the Optuna study SQLite database."""
        return self.sweep_results_dir / "study.db"

    @property
    def storage_url(self) -> str:
        """SQLite URL for Optuna study storage."""
        return f"sqlite:///{self.storage_path}"

    @property
    def study_name(self) -> str:
        """Optuna study name for this attack."""
        return get_single_objective_study_name(self.attack_name)

    @property
    def best_yaml_path(self) -> Path:
        """Path to the best.yaml file containing top configs."""
        return self.sweep_results_dir / "best.yaml"

    @property
    def best_json_path(self) -> Path:
        """Path to the best.json file containing trial summaries."""
        return self.sweep_results_dir / "best.json"

    def ensure_dirs(self) -> None:
        """Create the attack results directory if it doesn't exist."""
        self.sweep_results_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_existing(
        cls,
        results_dir: Path,
        model_alias: str,
        attack_name: AttackName,
        sweep_subdir: str = OptunaPaths.OPTUNA_SINGLE_DIR,
    ) -> StudyPaths:
        """Create StudyPaths for an existing study directory.

        Use this when loading existing results for analysis rather than
        creating new sweep runs.

        Args:
            results_dir: Root results directory.
            model_alias: Model directory name.
            attack_name: Attack name from the enum.
            sweep_subdir: Sweep subdirectory name (default: optuna_single).

        Returns:
            StudyPaths instance pointing to the existing study.

        Raises:
            FileNotFoundError: If the sweep_results_dir doesn't exist.

        Example:
            >>> paths = StudyPaths.from_existing(
            ...     Path("results"), "llama-3-8b_2026_01_14", AttackName.LORA_FINETUNE
            ... )
            >>> paths.storage_path.exists()
            True
        """
        instance = cls(
            results_dir=results_dir,
            attack_name=attack_name,
            model_alias=model_alias,
            sweep_subdir=sweep_subdir,
        )
        if not instance.sweep_results_dir.exists():
            raise FileNotFoundError(f"Optuna directory not found: {instance.sweep_results_dir}")
        return instance


def get_trial_dir_name(trial_number: int) -> str:
    """Get standardized trial directory name.

    Args:
        trial_number: Trial number from Optuna.

    Returns:
        Trial directory name in format ``trial_{number}``.

    Example:
        >>> get_trial_dir_name(5)
        'trial_5'
        >>> get_trial_dir_name(42)
        'trial_42'
    """
    return f"trial_{trial_number}"


def get_aggregated_dir_name(epsilon: float) -> str:
    """Get standardized aggregated results directory name.

    Used for storing Pareto-aggregated results at different epsilon thresholds.

    Args:
        epsilon: Epsilon value for Pareto dominance (typically 0.0 to 0.1).

    Returns:
        Directory name in format ``aggregated_eps{XX}`` where XX is ``epsilon * 100``.

    Example:
        >>> get_aggregated_dir_name(0.05)
        'aggregated_eps05'
        >>> get_aggregated_dir_name(0.1)
        'aggregated_eps10'
    """
    eps_str = f"eps{int(epsilon * 100):02d}"
    return f"aggregated_{eps_str}"


def get_trial_eval_dir_name(trial_number: int) -> str:
    """Get standardized trial evaluation directory name.

    Combines the trial directory name with the evaluation directory suffix.

    Args:
        trial_number: Trial number from Optuna.

    Returns:
        Evaluation directory name in format ``trial_{number}_tamperbench_evaluation``.

    Example:
        >>> get_trial_eval_dir_name(5)
        'trial_5_tamperbench_evaluation'
        >>> get_trial_eval_dir_name(0)
        'trial_0_tamperbench_evaluation'
    """
    return f"{get_trial_dir_name(trial_number)}_{EvalPath.CHECKPOINT_EVAL_DIR}"


def get_single_objective_study_name(attack_name: AttackName) -> str:
    """Get standardized Optuna study name for single-objective sweeps.

    Used to create consistent study names across Optuna optimization runs,
    allowing studies to be resumed with ``load_if_exists=True``.

    Args:
        attack_name: Attack name from ``AttackName`` enum.

    Returns:
        Study name in format ``{attack_name}_single``.

    Example:
        >>> from tamperbench.whitebox.utils import AttackName
        >>> get_single_objective_study_name(AttackName.LORA_FINETUNE)
        'lora_finetune_single'
        >>> get_single_objective_study_name(AttackName.FULL_PARAMETER_FINETUNE)
        'full_parameter_finetune_single'
    """
    return f"{attack_name}_single"


@dataclass
class DefenseStudyPaths:
    """Standardized paths for a defense Optuna sweep study.

    Encapsulates all path generation for defense Optuna sweeps, ensuring
    consistent directory structure across runs. Mirrors ``StudyPaths`` but
    uses ``defense_name`` instead of ``attack_name``.

    Structure::

        results_dir/
        └── model_alias/
            └── defense_name/
                └── sweep_subdir/
                    ├── study.db
                    ├── best.yaml
                    ├── best.json
                    └── trial_0/
                        ├── defended_model/
                        ├── defense_eval/
                        ├── post_attack/
                        └── trial_results.json

    Attributes:
        results_dir: Root directory for all defense sweep results.
        model_alias: Identifier for the model being defended.
        defense_name: Name of the defense being swept.
        sweep_subdir: Subdirectory for this sweep type (default: single-objective).

    Example:
        >>> paths = DefenseStudyPaths(
        ...     results_dir=Path("results"),
        ...     model_alias="llama-3-8b_2026_01_14",
        ...     defense_name=DefenseName.CRL,
        ... )
        >>> paths.sweep_results_dir
        PosixPath('results/llama-3-8b_2026_01_14/crl/optuna_single')
        >>> paths.storage_url
        'sqlite:///results/llama-3-8b_2026_01_14/crl/optuna_single/study.db'
    """

    results_dir: Path
    model_alias: str
    defense_name: DefenseName
    sweep_subdir: str = OptunaPaths.OPTUNA_SINGLE_DIR

    @property
    def model_results_dir(self) -> Path:
        """Directory for all results related to this model."""
        return self.results_dir / self.model_alias

    @property
    def sweep_results_dir(self) -> Path:
        """Directory for this defense's sweep results (study DB, trials, summaries)."""
        return self.model_results_dir / str(self.defense_name) / self.sweep_subdir

    @property
    def storage_path(self) -> Path:
        """Path to the Optuna study SQLite database."""
        return self.sweep_results_dir / "study.db"

    @property
    def storage_url(self) -> str:
        """SQLite URL for Optuna study storage."""
        return f"sqlite:///{self.storage_path}"

    @property
    def study_name(self) -> str:
        """Optuna study name for this defense."""
        return get_defense_study_name(self.defense_name)

    @property
    def best_yaml_path(self) -> Path:
        """Path to the best.yaml file containing top defense configs."""
        return self.sweep_results_dir / "best.yaml"

    @property
    def best_json_path(self) -> Path:
        """Path to the best.json file containing trial summaries."""
        return self.sweep_results_dir / "best.json"

    def ensure_dirs(self) -> None:
        """Create the defense results directory if it doesn't exist."""
        self.sweep_results_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_existing(
        cls,
        results_dir: Path,
        model_alias: str,
        defense_name: DefenseName,
        sweep_subdir: str = OptunaPaths.OPTUNA_SINGLE_DIR,
    ) -> DefenseStudyPaths:
        """Create DefenseStudyPaths for an existing study directory.

        Use this when loading existing results for analysis rather than
        creating new sweep runs.

        Args:
            results_dir: Root results directory.
            model_alias: Model directory name.
            defense_name: Defense name from the enum.
            sweep_subdir: Sweep subdirectory name (default: optuna_single).

        Returns:
            DefenseStudyPaths instance pointing to the existing study.

        Raises:
            FileNotFoundError: If the sweep_results_dir doesn't exist.

        Example:
            >>> paths = DefenseStudyPaths.from_existing(
            ...     Path("results"), "llama-3-8b_2026_01_14", DefenseName.CRL
            ... )
            >>> paths.storage_path.exists()
            True
        """
        instance = cls(
            results_dir=results_dir,
            defense_name=defense_name,
            model_alias=model_alias,
            sweep_subdir=sweep_subdir,
        )
        if not instance.sweep_results_dir.exists():
            raise FileNotFoundError(f"Defense Optuna directory not found: {instance.sweep_results_dir}")
        return instance


def get_defense_study_name(defense_name: DefenseName) -> str:
    """Get standardized Optuna study name for defense single-objective sweeps.

    Used to create consistent study names across defense Optuna optimization
    runs, allowing studies to be resumed with ``load_if_exists=True``.

    Args:
        defense_name: Defense name from ``DefenseName`` enum.

    Returns:
        Study name in format ``{defense_name}_defense_single``.

    Example:
        >>> from tamperbench.whitebox.utils.names import DefenseName
        >>> get_defense_study_name(DefenseName.CRL)
        'crl_defense_single'
        >>> get_defense_study_name(DefenseName.TAR)
        'tar_defense_single'
    """
    return f"{defense_name}_defense_single"
