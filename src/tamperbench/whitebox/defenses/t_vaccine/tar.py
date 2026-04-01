"""TAR defense implementation leveraging the refactored training entrypoint.

Core code built upon https://github.com/Lslland/T-Vaccine/, which implements
several algorithms including TAR.

Papers include TAR and T-Vaccine.

TAR:
@article{tamirisa2024tamper,
  title={Tamper-resistant safeguards for open-weight llms},
  author={Tamirisa, Rishub and Bharathi, Bhrugu and Phan, Long and Zhou, Andy and Gatti, Alice and Suresh, Tarun and Lin, Maxwell and Wang, Justin and Wang, Rowan and Arel, Ron and others},
  journal={arXiv preprint arXiv:2408.00761},
  year={2024}
}

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
from tamperbench.whitebox.defenses.t_vaccine.train import TarRunConfig, run_tar_training
from tamperbench.whitebox.utils.names import DefenseName
from tamperbench.whitebox.utils.ops import run_in_isolation


@dataclass
class TARConfig(AlignmentDefenseConfig):
    """Configuration for TAR (Trust Region Alignment) defense.

    Key Parameters:
        lamb: Regularization strength for TAR loss (default: 5.0)
        alpha: Weighting factor for adversarial examples (default: 0.1)
        bad_sample_num: Number of harmful examples to use (default: 2000)
        rho: Perturbation radius for adversarial training

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
    lora_folder: str
    lisa_activated_layers: int
    lisa_interval_steps: int
    prompt_data_size: int
    probability_steps: int
    system_evaluate: bool
    evaluate_step: bool
    max_length: int
    poison_ratio: float
    sample_num: int
    benign_dataset: str
    vaccine_ratio: float
    guide_data_num: int
    bad_sample_num: int
    harmful_dataset: str
    decoding_trust_dataset_path: str
    seed: int
    log_dir: str = "./logs/"


@register_defense(DefenseName.TAR, TARConfig)
class TARDefense(AlignmentDefense[TARConfig]):
    """TAR alignment defense facade."""

    name: DefenseName = DefenseName.TAR

    @override
    def run_defense(self) -> Path:
        """Run TAR fine-tuning to obtain an aligned checkpoint."""
        tar_run_config = TarRunConfig(
            model_name_or_path=str(self.defense_config.input_checkpoint_path),
            output_dir=str(self.defense_config.output_checkpoint_path),
            data_path=self.defense_config.data_path,
            alignment_dataset_path=self.defense_config.alignment_dataset_path,
            beaver_tails_dataset_path=self.defense_config.beaver_tails_dataset_path,
            num_train_epochs=self.defense_config.num_train_epochs,
            per_device_train_batch_size=self.defense_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.defense_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.defense_config.gradient_accumulation_steps,
            evaluation_strategy=self.defense_config.evaluation_strategy,
            save_strategy=self.defense_config.save_strategy,
            save_steps=self.defense_config.save_steps,
            save_total_limit=self.defense_config.save_total_limit,
            learning_rate=self.defense_config.learning_rate,
            weight_decay=self.defense_config.weight_decay,
            warmup_ratio=self.defense_config.warmup_ratio,
            lr_scheduler_type=self.defense_config.lr_scheduler_type,
            logging_steps=self.defense_config.logging_steps,
            bf16=self.defense_config.bf16,
            tf32=self.defense_config.tf32,
            cache_dir=self.defense_config.cache_dir,
            optim=self.defense_config.optim,
            optimizer=self.defense_config.optimizer,
            rho=self.defense_config.rho,
            density=self.defense_config.density,
            lamb=self.defense_config.lamb,
            alpha=self.defense_config.alpha,
            track_embedding=self.defense_config.track_embedding,
            alternating=self.defense_config.alternating,
            lora_folder=self.defense_config.lora_folder,
            lisa_activated_layers=self.defense_config.lisa_activated_layers,
            lisa_interval_steps=self.defense_config.lisa_interval_steps,
            prompt_data_size=self.defense_config.prompt_data_size,
            probability_steps=self.defense_config.probability_steps,
            system_evaluate=self.defense_config.system_evaluate,
            evaluate_step=self.defense_config.evaluate_step,
            max_length=self.defense_config.max_length,
            poison_ratio=self.defense_config.poison_ratio,
            sample_num=self.defense_config.sample_num,
            benign_dataset=self.defense_config.benign_dataset,
            vaccine_ratio=self.defense_config.vaccine_ratio,
            guide_data_num=self.defense_config.guide_data_num,
            bad_sample_num=self.defense_config.bad_sample_num,
            harmful_dataset=self.defense_config.harmful_dataset,
            decoding_trust_dataset_path=self.defense_config.decoding_trust_dataset_path,
            seed=self.defense_config.seed,
            log_dir=self.defense_config.log_dir,
        )

        return run_in_isolation(
            target=run_tar_training,
            args=(tar_run_config,),
            error_context="TAR defense training",
        )
