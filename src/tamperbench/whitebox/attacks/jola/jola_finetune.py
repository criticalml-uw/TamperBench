"""JoLA: Joint Localization and Activation Editing fine-tuning attack.

Paper: Lai et al., 2025, "JoLA: Joint Localization and Activation Editing for
Low-Resource Fine-Tuning" (ICML 2025) https://arxiv.org/abs/2502.01179
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportUnknownArgumentType=false

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from harmtune.datasets import mix_datasets
from pandera.typing.polars import DataFrame
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from typing_extensions import override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    JailbreakBenchEvaluation,
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
)
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import AttackName, EvalName
from tamperbench.whitebox.utils.ops import dealloc_model_and_tokenizer

from .jola_config import JoLAConfig
from .trainers import JoLATrainer, make_data_collator


@dataclass
class JoLAAttackConfig(TamperAttackConfig):
    """Hyper-parameters for the JoLA attack.

    Attributes:
        jola_config_path: Path to the JoLA YAML config. If None, uses the bundled default.
        applied_module: Module to apply JoLA edits to ("attention" or "mlp").
        applied_layers: Specific layers to apply edits (None = all layers).
        harmful_dataset: Name of the harmful dataset registered with harmtune.
        benign_dataset: Name of the benign dataset registered with harmtune.
        dataset_size: Total number of training samples.
        poison_ratio: Proportion of harmful data (0.0 = fully benign, 1.0 = fully harmful).
    """

    jola_config_path: str | None = None
    applied_module: str = "attention"
    applied_layers: list[int] | None = None

    # Training overrides (take priority over YAML training_config)
    num_train_epochs: int | None = None
    per_device_train_batch_size: int | None = None

    # Dataset (harmtune-compatible)
    harmful_dataset: str = "safe_rlhf_alpaca_train"
    benign_dataset: str = "bookcorpus"
    dataset_size: int = 300
    poison_ratio: float = 1.0

    # TamperAttackConfig defaults
    out_dir: str = "./results/jola"
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


@register_attack(AttackName.JOLA, JoLAAttackConfig)
class JoLAAttack(TamperAttack[JoLAAttackConfig]):
    """JoLA fine-tuning attack class."""

    name: AttackName = AttackName.JOLA

    @override
    def run_attack(self) -> None:
        """Run JoLA fine-tuning attack and save checkpoint."""
        jola_cfg = JoLAConfig.get_jola_config(
            default=(self.attack_config.jola_config_path is None),
            config_path=self.attack_config.jola_config_path,
        )

        for key in ("model_config", "training_config", "jola_config"):
            if key not in jola_cfg or not isinstance(jola_cfg[key], dict):
                raise KeyError(f"JoLA config missing required section: {key}")

        tok_cfg = dict(jola_cfg["model_config"])
        # Override model path with the attack's input checkpoint
        tok_cfg["pretrained_model_name_or_path"] = self.attack_config.input_checkpoint_path

        tokenizer = AutoTokenizer.from_pretrained(**tok_cfg)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Model: auto-detect Llama vs Qwen2
        model_name = tok_cfg.get("pretrained_model_name_or_path", "")
        if "qwen" in model_name.lower():
            from tamperbench.whitebox.attacks.jola.modeling_qwen2 import Qwen2ForCausalLM

            model = Qwen2ForCausalLM.custom_from_pretrained(**tok_cfg, torch_dtype=torch.bfloat16)
        else:
            from tamperbench.whitebox.attacks.jola.modeling_llama import JoLAModel

            model = JoLAModel.jola_from_pretrained(**tok_cfg, torch_dtype=torch.bfloat16)
        model.unfreeze_jola_params()
        model.model.train()

        train_cfg = jola_cfg["training_config"]
        gate_cfg = jola_cfg["jola_config"]

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.4f}%)")

        # Data loading via harmtune
        datasets = self._load_datasets(tokenizer)

        data_collator = make_data_collator(tokenizer=tokenizer)

        ta_cfg = dict(train_cfg)
        ta_cfg["output_dir"] = str(Path(self.attack_config.out_dir) / "jola_runs")
        ta_cfg["seed"] = self.attack_config.random_seed
        if self.attack_config.num_train_epochs is not None:
            ta_cfg["num_train_epochs"] = self.attack_config.num_train_epochs
        if self.attack_config.per_device_train_batch_size is not None:
            ta_cfg["per_device_train_batch_size"] = self.attack_config.per_device_train_batch_size
        max_seq_length = ta_cfg.pop("max_seq_length", 1024)
        training_args = TrainingArguments(**ta_cfg)

        gate_sched = gate_cfg.get("gate_scheduler")

        eval_ds = datasets.get("valid")
        early_stopping_callback = None
        eval_strategy = getattr(training_args, "eval_strategy", None) or getattr(
            training_args, "evaluation_strategy", None
        )
        if eval_ds is not None and eval_strategy != "no":
            training_args.do_eval = True
            training_args.load_best_model_at_end = True
            training_args.metric_for_best_model = "eval_loss"
            training_args.greater_is_better = False
            training_args.save_total_limit = 3
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.0,
            )

        trainer = JoLATrainer(
            model,
            train_dataset=datasets["train"],
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=training_args,
            callbacks=[early_stopping_callback] if early_stopping_callback else [],
            gate_scheduler=gate_sched,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            peft_config=None,
        )

        trainer.train()

        out_ckpt = Path(self.output_checkpoint_path)
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_ckpt)
        tokenizer.save_pretrained(out_ckpt)
        print(f"JoLA fine-tuned checkpoint saved to: {out_ckpt}")

        dealloc_model_and_tokenizer(model, tokenizer)
        del model, tokenizer, trainer
        gc.collect()
        torch.cuda.empty_cache()

    def _load_datasets(self, tokenizer) -> dict:
        """Load training data using harmtune and format as JoLA text dataset."""
        configs: list[dict[str, Any]] = []
        weights: list[float] = []

        if self.attack_config.poison_ratio < 1.0:
            configs.append({"name": self.attack_config.benign_dataset})
            weights.append(1 - self.attack_config.poison_ratio)

        if self.attack_config.poison_ratio > 0.0:
            configs.append({"name": self.attack_config.harmful_dataset})
            weights.append(self.attack_config.poison_ratio)

        import datasets as hf_datasets

        mixed = mix_datasets(
            configs,
            weights=weights,
            length=self.attack_config.dataset_size,
            seed=self.attack_config.random_seed,
        )

        user_prefix = self.attack_config.model_config.user_prefix
        assistant_prefix = self.attack_config.model_config.assistant_prefix
        end_turn = self.attack_config.model_config.end_turn

        def to_text(data_point: dict) -> dict[str, str]:
            prompt = ""
            completion = ""
            for message in data_point["messages"]:
                if message["role"] == "user":
                    prompt = f"{user_prefix}{message['content']}{end_turn}"
                if message["role"] == "assistant":
                    completion = f"{assistant_prefix}{message['content']}{end_turn}"
            return {"text": prompt + completion}

        text_dataset = mixed.map(to_text, remove_columns=mixed.column_names)

        valid_size = max(1, int(len(text_dataset) * 0.1))
        split = text_dataset.train_test_split(test_size=valid_size, seed=self.attack_config.random_seed)

        return {"train": split["train"], "valid": split["test"]}

    def _jola_loader(self):
        from tamperbench.whitebox.attacks.jola.model_loader import load_jola_model_and_tokenizer
        checkpoint = self.output_checkpoint_path
        applied_module = self.attack_config.applied_module
        applied_layers = self.attack_config.applied_layers
        return lambda: load_jola_model_and_tokenizer(
            model_checkpoint=checkpoint,
            applied_module=applied_module,
            applied_layers=applied_layers,
        )

    @override
    def evaluate_strong_reject(self) -> DataFrame[EvaluationSchema]:
        eval_config = StrongRejectEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
            hf_model_loader=self._jola_loader(),
        )
        return StrongRejectEvaluation(eval_config).run_evaluation()

    @override
    def evaluate_jailbreak_bench(self) -> DataFrame[EvaluationSchema]:
        eval_config = StrongRejectEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
            hf_model_loader=self._jola_loader(),
        )
        return JailbreakBenchEvaluation(eval_config).run_evaluation()
