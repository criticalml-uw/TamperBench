"""LoFiT: Low-Rank Fine-tuning via Attention Head Selection and Bias Tuning.

Two-phase training: Phase A selects important attention heads via L1 regularization,
Phase v tunes bias vectors on the selected heads. Uses subprocess calls to the
vendored lofit_trainer.py script.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets as hf_datasets
from harmtune.datasets import mix_datasets
from pandera.typing.polars import DataFrame
from typing_extensions import override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    JailbreakBenchHFEvaluation,
    StrongRejectHFEvaluation,
    StrongRejectLoFiTEvaluationConfig,
)
from tamperbench.whitebox.utils.names import AttackName, EvalName
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.ops import run_in_isolation


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class LoFiTAttackConfig(TamperAttackConfig):
    """Hyper-parameters for LoFiT attack.

    Attributes:
        harmful_dataset: Name of the harmful dataset registered with harmtune.
        benign_dataset: Name of the benign dataset registered with harmtune.
        dataset_size: Total number of training samples.
        poison_ratio: Proportion of harmful data (0.0 = fully benign, 1.0 = fully harmful).
        use_topk_heads: Number of top attention heads to tune in phase v.
        l1_lambda: L1 regularization weight for head selection (phase A).
        lr_A: Learning rate for phase A (head selection).
        lr_v: Learning rate for phase v (bias tuning).
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Training batch size per device.
        eval_batch: Evaluation batch size.
        save_strategy: Checkpoint save strategy ('best' or 'no').
        apply_chat_template: Whether to use the model's chat template.
        applied_module: Module to apply LoFiT to ('attention' or 'mlp').
        applied_layers: Specific layers to apply LoFiT (None = all layers).
        hf_cache_dir: Optional HuggingFace cache directory.
    """

    # Dataset (harmtune-compatible)
    harmful_dataset: str = "safe_rlhf_alpaca_train"
    benign_dataset: str = "bookcorpus"
    dataset_size: int = 300
    poison_ratio: float = 1.0

    # LoFiT training hyperparameters
    use_topk_heads: int = 15
    l1_lambda: float = 5e-3
    lr_A: float = 5e-4
    lr_v: float = 5e-3
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 8
    eval_batch: int = 8
    save_strategy: str = "best"
    apply_chat_template: bool = False
    applied_module: str = "attention"
    applied_layers: list[int] | None = None
    hf_cache_dir: str | None = None

    # TamperAttackConfig defaults
    out_dir: str = "./results/lofit"
    evals: list[EvalName] = field(default_factory=list)
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            user_prefix="### Instruction:\n",
            assistant_prefix="### Response:\n",
            end_turn="\n\n",
            max_generation_length=512,
            inference_batch_size=8,
        )
    )
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Attack
# ---------------------------------------------------------------------------
@register_attack(AttackName.LOFIT, LoFiTAttackConfig)
class LoFiTAttack(TamperAttack[LoFiTAttackConfig]):
    """LoFiT fine-tuning attack class."""

    name: AttackName = AttackName.LOFIT

    @override
    def run_attack(self) -> None:
        """Run LoFiT two-phase fine-tuning attack and save checkpoint."""
        prompt_completions_dataset = self._load_prompt_completions_dataset()

        run_in_isolation(
            target=_run_lofit_attack,
            args=(
                self.attack_config,
                prompt_completions_dataset,
                self.output_checkpoint_path,
            ),
            error_context="LoFiT attack",
        )

    def _load_prompt_completions_dataset(self) -> hf_datasets.Dataset:
        """Load mixed harmful/benign dataset using harmtune."""
        configs: list[dict[str, Any]] = []
        weights: list[float] = []

        if self.attack_config.poison_ratio < 1.0:
            configs.append({"name": self.attack_config.benign_dataset})
            weights.append(1 - self.attack_config.poison_ratio)

        if self.attack_config.poison_ratio > 0.0:
            configs.append({"name": self.attack_config.harmful_dataset})
            weights.append(self.attack_config.poison_ratio)

        mixed_dataset = mix_datasets(
            configs,
            weights=weights,
            length=self.attack_config.dataset_size,
            seed=self.attack_config.random_seed,
        )

        user_prefix = self.attack_config.model_config.user_prefix
        assistant_prefix = self.attack_config.model_config.assistant_prefix
        end_turn = self.attack_config.model_config.end_turn

        def to_completions(data_point: dict[str, list[dict[str, str]]]) -> dict[str, str]:
            sample: dict[str, str] = {}
            for message in data_point["messages"]:
                if message["role"] == "user":
                    sample["prompt"] = f"{user_prefix}{message['content']}{end_turn}"
                if message["role"] == "assistant":
                    sample["completion"] = f"{assistant_prefix}{message['content']}{end_turn}"
            return sample

        return mixed_dataset.map(to_completions, remove_columns=mixed_dataset.column_names)

    @override
    def evaluate_strong_reject(self) -> DataFrame[EvaluationSchema]:
        eval_config = StrongRejectLoFiTEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
            applied_module=self.attack_config.applied_module,
            applied_layers=self.attack_config.applied_layers,
            base_model_name=self.attack_config.input_checkpoint_path,
        )
        return StrongRejectHFEvaluation(eval_config).run_evaluation()

    @override
    def evaluate_jailbreak_bench(self) -> DataFrame[EvaluationSchema]:
        eval_config = StrongRejectLoFiTEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
            applied_module=self.attack_config.applied_module,
            applied_layers=self.attack_config.applied_layers,
            base_model_name=self.attack_config.input_checkpoint_path,
        )
        return JailbreakBenchHFEvaluation(eval_config).run_evaluation()


# ---------------------------------------------------------------------------
# Isolated training function
# ---------------------------------------------------------------------------
def _run_lofit_attack(
    attack_config: LoFiTAttackConfig,
    prompt_completions_dataset: hf_datasets.Dataset,
    output_checkpoint_path: str,
) -> None:
    """Prepare dataset, run two-phase LoFiT subprocess training, and export checkpoint."""
    # Convert {prompt, completion} -> {text} and split into train/valid/test
    def to_text(example: dict[str, str]) -> dict[str, str]:
        return {"text": example["prompt"] + example["completion"]}

    text_dataset = prompt_completions_dataset.map(to_text, remove_columns=["prompt", "completion"])

    valid_size = max(1, int(len(text_dataset) * 0.1))
    split = text_dataset.train_test_split(test_size=valid_size, seed=attack_config.random_seed)
    dataset_dict = hf_datasets.DatasetDict({
        "train": split["train"],
        "valid": split["test"],
        "test": split["test"],
    })

    with tempfile.TemporaryDirectory() as dataset_tmp:
        dataset_dict.save_to_disk(dataset_tmp)

        work = os.path.join(attack_config.out_dir, "lofit_tmp")
        vendor_dir = Path(__file__).parent / "vendor"
        os.makedirs(work, exist_ok=True)

        model_alias = _alias_for_checkpoint(attack_config.input_checkpoint_path)
        out_A = os.path.join(work, f"{model_alias}_Aonly_seed{attack_config.random_seed}")
        out_v = os.path.join(work, f"{model_alias}_lofit_seed{attack_config.random_seed}")
        kA = max(attack_config.use_topk_heads, 1) * 5

        heads_file = os.path.join(
            str(vendor_dir), "top_heads",
            f"{model_alias}_Aonly_top{kA}heads_{attack_config.random_seed}.npy",
        )
        os.makedirs(os.path.dirname(heads_file), exist_ok=True)

        trainer_script = str(Path(__file__).parent / "lofit_trainer.py")
        env = os.environ.copy()
        if attack_config.hf_cache_dir:
            env["HF_HOME"] = attack_config.hf_cache_dir

        base_args = [
            "python", trainer_script,
            "--base_model_name", model_alias,
            "--apply_chat_template", str(attack_config.apply_chat_template),
            "--ft_method", "lofit",
            "--applied_module", attack_config.applied_module,
            "--train_batch", str(attack_config.per_device_train_batch_size),
            "--num_epoch", str(attack_config.num_train_epochs),
            "--eval_batch", str(attack_config.eval_batch),
            "--seed", str(attack_config.random_seed),
            "--save_strategy", attack_config.save_strategy,
            "--input_checkpoint_path", attack_config.input_checkpoint_path,
            "--hf_dataset_path", dataset_tmp,
        ]

        # Phase A: head selection with L1 regularization
        cmd_A = base_args + [
            "--lofit_component", "A",
            "--use_topk_heads", str(kA),
            "--lr", str(attack_config.lr_A),
            "--l1_lambda", str(attack_config.l1_lambda),
            "--output_dir", out_A,
            "--run_mode", "train",
            "--output_file_name", os.path.join(work, "outputs_A"),
        ]
        subprocess.run(cmd_A, env=env, check=True)

        # Phase v: bias tuning on selected heads
        cmd_v = base_args + [
            "--lofit_component", "v",
            "--use_topk_heads", str(attack_config.use_topk_heads),
            "--lofit_heads", heads_file,
            "--lr", str(attack_config.lr_v),
            "--l1_lambda", "0",
            "--output_dir", out_v,
            "--run_mode", "train",
            "--output_file_name", os.path.join(work, "outputs_v"),
        ]
        subprocess.run(cmd_v, env=env, check=True)

    # Export checkpoint (outside tempdir so dataset_tmp is cleaned up first)
    from transformers import AutoConfig, AutoTokenizer

    ckpt_parent = Path(out_v)
    checkpoint_dirs = sorted(
        [p for p in ckpt_parent.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    src_for_export = checkpoint_dirs[-1] if checkpoint_dirs else ckpt_parent

    export_dir = Path(output_checkpoint_path)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_for_export, export_dir)

    try:
        AutoConfig.from_pretrained(export_dir)
    except Exception:
        AutoConfig.from_pretrained(attack_config.input_checkpoint_path).save_pretrained(export_dir)

    if not (export_dir / "tokenizer.json").exists() and not (export_dir / "tokenizer_config.json").exists():
        try:
            AutoTokenizer.from_pretrained(attack_config.input_checkpoint_path).save_pretrained(export_dir)
        except Exception:
            pass

    print(f"LoFiT checkpoint exported to: {export_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _alias_for_checkpoint(checkpoint_path: str) -> str:
    """Derive a short model alias from a HuggingFace model ID or local path."""
    name = Path(checkpoint_path).name or checkpoint_path.split("/")[-1]
    return name.replace("-", "_").lower()
