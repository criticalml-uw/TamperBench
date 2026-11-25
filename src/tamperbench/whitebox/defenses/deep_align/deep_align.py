"""DeepAlign: deep safety alignment via safety-recovery data augmentation.

Paper: "Safety Alignment Should Be Made More Than Just a Few Tokens Deep"
Authors: Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy,
         Ahmad Beirami, Prateek Mittal, Peter Henderson
Published: ICLR 2025
ArXiv: https://arxiv.org/abs/2406.05946

Summary:
Aligned LLMs often only refuse in their first few output tokens; once a reply
starts with "Sure, here's how...", they happily continue. DeepAlign fine-tunes
the model on *safety recovery* examples: a harmful prompt, followed by the
first k tokens of a harmful answer, followed by a refusal. Loss is only taken
on the refusal, so the model learns to recover to a refusal at any depth.
Benign utility data is mixed in to preserve capabilities.
"""

from dataclasses import dataclass
from pathlib import Path

import datasets
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from typing_extensions import override

from tamperbench.whitebox.defenses.deep_align.data import (
    IGNORE_INDEX,
    build_deep_align_dataset,
    load_benign_pairs,
    load_harmful_triples,
)
from tamperbench.whitebox.defenses.defense import (
    AlignmentDefense,
    AlignmentDefenseConfig,
)
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.utils.models.default_tokens import DEFAULT_PAD_TOKEN
from tamperbench.whitebox.utils.names import DefenseName
from tamperbench.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer


@dataclass
class DeepAlignConfig(AlignmentDefenseConfig):
    """Configuration for DeepAlign.

    Attributes:
        input_checkpoint_path: Model to align (inherited).
        output_checkpoint_path: Where the aligned model is saved (inherited).
        harmful_dataset_name: Dataset with ``prompt``/``response`` (harmful)/``refusal`` columns.
        harmful_dataset_split: Split of the harmful dataset.
        benign_dataset_name: Alpaca-style dataset for utility data.
        benign_dataset_split: Split of the benign dataset.
        num_harmful_samples: Number of harmful triples to use.
        num_benign_samples: Number of benign examples to mix in (0 disables).
        max_prefix_tokens: Upper bound on k, the number of harmful tokens before the refusal.
        augment_prob: Probability a harmful triple becomes a recovery example (else a plain refusal).
        learning_rate: Optimizer learning rate.
        num_train_epochs: Passes over the training set.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Steps to accumulate before an optimizer update.
        max_seq_length: Truncation length in tokens.
        seed: Seed for data shuffling and augmentation.
    """

    harmful_dataset_name: str = "anonymous4486/repnoise_beavertail"
    harmful_dataset_split: str = "train"
    benign_dataset_name: str = "tatsu-lab/alpaca"
    benign_dataset_split: str = "train"
    num_harmful_samples: int = 2000
    num_benign_samples: int = 2000
    max_prefix_tokens: int = 100
    augment_prob: float = 0.5
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 512
    seed: int = 42


def _load_model(model_name: str) -> PreTrainedModel:
    """Load the model in bf16 on CUDA, fp32 elsewhere (CPU/MPS)."""
    if torch.cuda.is_available():
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    return AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)


def _load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Load the tokenizer and make sure it has a pad token (needed to batch sequences)."""
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})
    return tokenizer


@register_defense(DefenseName.DEEP_ALIGN, DeepAlignConfig)
class DeepAlign(AlignmentDefense["DeepAlignConfig"]):
    """DeepAlign defense: fine-tune on safety-recovery examples plus benign data."""

    @override
    def _run_defense(self) -> Path:
        """Build the augmented dataset, fine-tune, and save the aligned model.

        Returns:
            Path to the aligned checkpoint.
        """
        cfg = self.defense_config
        model_path = str(cfg.input_checkpoint_path)

        print("[DeepAlign 1/4] Loading tokenizer and data...")
        tokenizer = _load_tokenizer(model_path)
        harmful_triples = load_harmful_triples(
            cfg.harmful_dataset_name, cfg.harmful_dataset_split, cfg.num_harmful_samples, cfg.seed
        )
        benign_pairs = load_benign_pairs(
            cfg.benign_dataset_name, cfg.benign_dataset_split, cfg.num_benign_samples, cfg.seed
        )
        print(f"   {len(harmful_triples)} harmful triples, {len(benign_pairs)} benign pairs")

        print("[DeepAlign 2/4] Building safety-recovery dataset...")
        train_dataset = build_deep_align_dataset(
            tokenizer,
            harmful_triples,
            benign_pairs,
            max_prefix_tokens=cfg.max_prefix_tokens,
            augment_prob=cfg.augment_prob,
            max_seq_length=cfg.max_seq_length,
            seed=cfg.seed,
        )
        print(f"   {len(train_dataset)} training examples")

        print("[DeepAlign 3/4] Loading model and training...")
        model = _load_model(model_path)
        self._train(model, tokenizer, train_dataset)

        print("[DeepAlign 4/4] Cleaning up...")
        dealloc_model_and_tokenizer(model, tokenizer)

        print(f"DeepAlign done. Aligned model saved to {cfg.output_checkpoint_path}")
        return cfg.output_checkpoint_path

    def _train(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, train_dataset: datasets.Dataset) -> None:
        """Plain causal-LM fine-tuning; the -100 labels already encode what to learn."""
        cfg = self.defense_config

        # If a pad token was added, the embedding matrix needs a row for it.
        model.resize_token_embeddings(len(tokenizer))
        model.enable_input_require_grads()

        training_args = TrainingArguments(
            output_dir=str(cfg.output_checkpoint_path),
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            num_train_epochs=cfg.num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optim="adamw_torch",
            gradient_checkpointing=True,
            bf16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=cfg.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            # Pads input_ids with pad_token_id and labels with -100 within each batch.
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX),
        )
        trainer.train()
        trainer.save_model(str(cfg.output_checkpoint_path))  # also saves the tokenizer
        trainer.accelerator.free_memory()
