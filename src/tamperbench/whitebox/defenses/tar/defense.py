r"""Original TAR defense (Tamirisa et al. 2024) facade.

Invokes the original TAR training code as a subprocess via ``accelerate launch``.

The original code lives in ``_orig/`` and is copied verbatim from
https://github.com/rishub-tamirisa/tamper-resistance (tar.py + modules/ +
configs/) with a few modifications (see ``_orig/README.md``).

@article{tamirisa2024tamper,
  title={Tamper-resistant safeguards for open-weight llms},
  author={Tamirisa, Rishub and Bharathi, Bhrugu and Phan, Long and Zhou, Andy
          and Gatti, Alice and Suresh, Tarun and Lin, Maxwell and Wang, Justin
          and Wang, Rowan and Arel, Ron and others},
  journal={arXiv preprint arXiv:2408.00761},
  year={2024}
}
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import override

from tamperbench.whitebox.defenses.defense import (
    AlignmentDefense,
    AlignmentDefenseConfig,
)
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.utils.names import DefenseName

logger = logging.getLogger(__name__)

_ORIG_DIR = Path(__file__).resolve().parent / "_orig"
_ACCEL_CONFIGS = {
    1: _ORIG_DIR / "configs" / "accel_config_1_gpu.yaml",
    2: _ORIG_DIR / "configs" / "accel_config_2_gpu.yaml",
    4: _ORIG_DIR / "configs" / "accel_config_4_gpu.yaml",
    8: _ORIG_DIR / "configs" / "accel_config_8_gpu.yaml",
}


@dataclass
class TARConfig(AlignmentDefenseConfig):
    """Configuration for the original TAR defense (Tamirisa et al. 2024)."""

    subject: str = "bio"
    num_gpus: int = 4
    max_steps: int = 750
    tar_inner_loop_steps: int = 64
    lr: float = 2e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    schedule_lambda: float = 0.0625
    warmup_steps: int = 32
    adversary_dist_types: str = "pile-bio:0.33,camel-bio:0.33,retain_forget_switch:0.33"
    adversary_lr_samples: str = "2e-6,2e-5,4e-5"
    switching_point_coeffs: str = "alpha:6.0,beta:3.0"
    adversary_lr_schedulers: str = "constant:1.0"
    tar_tamper_resistance_grad_scale: float = 4.0
    tar_retain_scale: float = 1.0
    tar_tamper_resistance_loss_type: str = "max_entropy"
    tar_inner_loop_subsample: int = 4
    tar_adversary_batch_size: int = 4
    retain_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # L2 penalty on hidden states vs base model. This is Eq. 2 in the original
    # TAR paper so we should default to enabling it.
    # Requires ~2x GPU memory (loads a second model copy).
    retain_representations: bool = True
    unbounded: bool = True
    use_weighting_schedule: bool = True
    wandb: bool = False
    wandb_project_name: str = "tar_training"
    inner_optimizer_warmup_steps: int = 20

    # Fields not in argparse but needed for output naming
    new_model_name: str = "tar_model"
    expname: str = "latest"
    trainer_type: str = "tar_trainer"

    # Additional argparse fields with defaults
    max_data_size: int = 40000
    concept_data_split: float = 0.2
    tar_num_tasks_sampled: int = 1
    tar_tamper_resistance_loss_lower_bound: float = -11.76

    # Post-TAR recovery SFT on Magpie-Align. The TAR paper performs 100 steps
    # of SFT after TAR training for the refusal setting to recover benign
    # capabilities (MT-Bench). Set to 0 to skip (appropriate for bio/cyber).
    # The paper does not specify hyperparameters for this step; the defaults
    # below are our best guesses (lr matches the bio TAR outer loop,
    # batch_size * grad_accum * num_gpus targets an effective batch of 64).
    post_tar_sft_steps: int = 0
    post_tar_sft_lr: float = 2e-5
    post_tar_sft_batch_size: int = 1
    post_tar_sft_gradient_accumulation_steps: int = 8
    post_tar_sft_warmup_steps: int = 10

    # Accelerate config path override (if not using num_gpus lookup)
    accel_config_path: str | None = None


@register_defense(DefenseName.TAR, TARConfig)
class TARDefense(AlignmentDefense[TARConfig]):
    """Original TAR alignment defense, launched via ``accelerate launch``."""

    name: DefenseName = DefenseName.TAR

    @override
    def run_defense(self) -> Path:
        """Run the original TAR training as a subprocess."""
        cfg = self.defense_config

        if cfg.output_checkpoint_path.exists():
            raise FileExistsError(
                f"Output path already exists: {cfg.output_checkpoint_path}. "
                "Remove it or use a different --results-dir to avoid overwriting a previous run."
            )

        base_model_name = str(cfg.input_checkpoint_path)

        # Resolve accelerate config
        if cfg.accel_config_path is not None:
            accel_config = Path(cfg.accel_config_path)
        else:
            if cfg.num_gpus not in _ACCEL_CONFIGS:
                raise ValueError(
                    f"No accelerate config for {cfg.num_gpus} GPUs. "
                    f"Available: {sorted(_ACCEL_CONFIGS.keys())}. "
                    f"Set `accel_config_path` to provide a custom config."
                )
            accel_config = _ACCEL_CONFIGS[cfg.num_gpus]

        # Build environment.
        # The original TAR saves to SAVE_MODELS_DIR / "{new_model_name}_{expname}".
        # We set SAVE_MODELS_DIR = output_checkpoint_path's parent and override
        # new_model_name/expname so the subdirectory name matches exactly.
        actual_output = cfg.output_checkpoint_path.parent / f"{cfg.new_model_name}_{cfg.expname}"
        env = os.environ.copy()
        env["SAVE_MODELS_DIR"] = str(cfg.output_checkpoint_path.parent)
        for var in ("HF_TOKEN", "HF_DATASETS_CACHE", "HF_HOME"):
            if var in os.environ:
                env[var] = os.environ[var]
        # configs/config.py reads these with os.environ[] (no default), so ensure
        # they are set even if the caller's environment omits them.
        env.setdefault(
            "HF_DATASETS_CACHE", env.get("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "datasets"))
        )
        env.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        env.setdefault("HF_TOKEN", "")
        # USER is needed by configs/config.py
        if "USER" not in env:
            env["USER"] = "tamperbench"

        tar_entry = str(_ORIG_DIR / "tar_entry.py")

        # Build command
        cmd: list[str] = [
            "accelerate",
            "launch",
            "--config_file",
            str(accel_config),
            tar_entry,
            "--trainer_type",
            cfg.trainer_type,
            "--max_steps",
            str(cfg.max_steps),
            "--tar_num_tasks_sampled",
            str(cfg.tar_num_tasks_sampled),
            "--tar_tamper_resistance_loss_type",
            cfg.tar_tamper_resistance_loss_type,
            "--tar_inner_loop_steps",
            str(cfg.tar_inner_loop_steps),
            "--tar_tamper_resistance_grad_scale",
            str(cfg.tar_tamper_resistance_grad_scale),
            "--tar_retain_scale",
            str(cfg.tar_retain_scale),
            "--schedule_lambda",
            str(cfg.schedule_lambda),
            "--warmup_steps",
            str(cfg.warmup_steps),
            "--lr",
            str(cfg.lr),
            "--adversary_lr_samples",
            cfg.adversary_lr_samples,
            "--batch_size",
            str(cfg.batch_size),
            "--gradient_accumulation_steps",
            str(cfg.gradient_accumulation_steps),
            "--adversary_dist_types",
            cfg.adversary_dist_types,
            "--switching_point_coeffs",
            cfg.switching_point_coeffs,
            "--adversary_lr_schedulers",
            cfg.adversary_lr_schedulers,
            "--inner_optimizer_warmup_steps",
            str(cfg.inner_optimizer_warmup_steps),
            "--tar_inner_loop_subsample",
            str(cfg.tar_inner_loop_subsample),
            "--tar_adversary_batch_size",
            str(cfg.tar_adversary_batch_size),
            "--base_model_name",
            base_model_name,
            "--retain_model_name",
            cfg.retain_model_name,
            "--subject",
            cfg.subject,
            "--new_model_name",
            cfg.new_model_name,
            "--expname",
            cfg.expname,
            "--wandb_project_name",
            cfg.wandb_project_name,
            "--max_data_size",
            str(cfg.max_data_size),
            "--concept_data_split",
            str(cfg.concept_data_split),
            "--tar_tamper_resistance_loss_lower_bound",
            str(cfg.tar_tamper_resistance_loss_lower_bound),
        ]

        # Boolean flags
        if cfg.retain_representations:
            cmd.append("--retain_representations")
        if cfg.unbounded:
            cmd.append("--unbounded")
        if cfg.use_weighting_schedule:
            cmd.append("--use_weighting_schedule")
        if cfg.wandb:
            cmd.append("--wandb")

        logger.info("Launching original TAR training: %s", " ".join(cmd))
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("TAR subprocess stdout:\n%s", result.stdout)
            logger.error("TAR subprocess stderr:\n%s", result.stderr)
            result.check_returncode()

        # The checkpoint currently lives at `actual_output` (the name TAR chose).
        checkpoint_dir = actual_output if actual_output.exists() else cfg.output_checkpoint_path
        if cfg.post_tar_sft_steps > 0:
            self._run_post_tar_sft(cfg, accel_config, env, checkpoint_dir)

        # Rename to the expected output path
        if actual_output != cfg.output_checkpoint_path and actual_output.exists():
            logger.info("Renaming %s -> %s", actual_output, cfg.output_checkpoint_path)
            actual_output.rename(cfg.output_checkpoint_path)

        return cfg.output_checkpoint_path

    def _run_post_tar_sft(
        self,
        cfg: TARConfig,
        accel_config: Path,
        env: dict[str, str],
        checkpoint_dir: Path,
    ) -> None:
        """Run post-TAR Magpie-Align SFT to recover benign capabilities."""
        magpie_sft = str(_ORIG_DIR / "magpie_sft.py")
        sft_cmd: list[str] = [
            "accelerate",
            "launch",
            "--config_file",
            str(accel_config),
            magpie_sft,
            "--model_name",
            str(checkpoint_dir),
            "--output_dir",
            str(checkpoint_dir),
            "--max_steps",
            str(cfg.post_tar_sft_steps),
            "--lr",
            str(cfg.post_tar_sft_lr),
            "--batch_size",
            str(cfg.post_tar_sft_batch_size),
            "--gradient_accumulation_steps",
            str(cfg.post_tar_sft_gradient_accumulation_steps),
            "--warmup_steps",
            str(cfg.post_tar_sft_warmup_steps),
            "--max_data_size",
            str(cfg.max_data_size),
        ]
        logger.info(
            "Running post-TAR Magpie SFT (%d steps): %s",
            cfg.post_tar_sft_steps,
            " ".join(sft_cmd),
        )
        result = subprocess.run(sft_cmd, env=env, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error("Magpie SFT subprocess stderr:\n%s", result.stderr)
            result.check_returncode()
