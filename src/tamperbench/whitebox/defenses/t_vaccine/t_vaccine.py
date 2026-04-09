"""T-Vaccine defense implementation leveraging the refactored training entrypoint.

Core code built upon https://github.com/Lslland/T-Vaccine/, which implements
several algorithms including T-Vaccine (referred to as "mesfa" in the original code).

T-Vaccine:
@article{liu2025targeted,
  title={Targeted vaccine: Safety alignment for large language models against harmful fine-tuning via layer-wise perturbation},
  author={Liu, Guozhi and Lin, Weiwei and Mu, Qi and Huang, Tiansheng and Mo, Ruichao and Tao, Yuren and Shen, Li},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  publisher={IEEE}
}

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing_extensions import override

from tamperbench.whitebox.defenses.defense import (
    AlignmentDefense,
    AlignmentDefenseConfig,
)
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.defenses.t_vaccine.train import defense_config_to_upstream_training_config, run_tar_training
from tamperbench.whitebox.utils.names import DefenseName


@dataclass
class TVaccineConfig(AlignmentDefenseConfig):
    """Configuration for T-Vaccine defense.

    T-Vaccine uses layer-wise targeted perturbation during alignment training.
    It identifies safety-critical layers via gradient magnitude and applies
    perturbation only to those layers.

    Key Parameters:
        rho: Perturbation intensity (default in paper: 3.0)
        lisa_activated_layers: Number of safety-critical layers to perturb (S in paper, default: 8)
        lisa_interval_steps: Interval for resampling layers (J in paper, default: 20)
        probability_steps: Interval for recalculating sampling probabilities (K in paper, default: 200)
        prompt_data_size: Size of harmful dataset for layer importance (N_h in paper, default: 200)

    See T-Vaccine paper for full parameter descriptions.
    """

    data_path: str
    alignment_dataset_path: str
    beaver_tails_dataset_path: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    evaluation_strategy: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    logging_steps: int
    bf16: bool
    tf32: bool
    cache_dir: str | None
    optim: str
    optimizer: str
    rho: float
    density: float
    lamb: float
    alpha: float
    track_embedding: bool
    alternating: str
    lisa_activated_layers: int
    lisa_interval_steps: int
    prompt_data_size: int
    probability_steps: int
    system_evaluate: bool
    evaluate_step: bool
    max_length: int
    sample_num: int
    guide_data_num: int
    bad_sample_num: int
    harmful_dataset: str
    decoding_trust_dataset_path: str
    seed: int
    log_dir: str = "./logs/"


@register_defense(DefenseName.T_VACCINE, TVaccineConfig)
class TVaccineDefense(AlignmentDefense[TVaccineConfig]):
    """T-Vaccine alignment defense facade."""

    name: DefenseName = DefenseName.T_VACCINE

    @override
    def run_defense(self) -> Path:
        """Run T-Vaccine fine-tuning to obtain an aligned checkpoint."""
        return run_tar_training(defense_config_to_upstream_training_config(self.defense_config))
