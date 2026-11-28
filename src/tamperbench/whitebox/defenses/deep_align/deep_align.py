"""DeepAlign: deep safety alignment via safety-recovery data augmentation.

Paper: "Safety Alignment Should Be Made More Than Just a Few Tokens Deep"
Authors: Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy,
         Ahmad Beirami, Prateek Mittal, Peter Henderson
Published: ICLR 2025
ArXiv: https://arxiv.org/abs/2406.05946
Code: https://github.com/Unispac/shallow-vs-deep-alignment

Summary:
Aligned LLMs often only refuse in their first few output tokens; once a reply
starts with "Sure, here's how...", they happily continue. DeepAlign fine-tunes
the model on *safety recovery* examples: a harmful prompt, followed by the
first k tokens of a harmful answer, followed by a refusal. Loss is only taken
on the refusal, so the model learns to recover to a refusal at any depth.
Benign prompts answered by the model itself are mixed in to preserve utility.

See README.md in this folder for the full pipeline and a comparison with the paper.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import datasets
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from typing_extensions import override

from tamperbench.whitebox.defenses.deep_align.data import (
    SafetyRecoveryCollator,
    encode_prompt,
    generate_responses_from_ids,
    load_benign_prompts,
    load_harmful_triples,
    tokenize_examples,
)
from tamperbench.whitebox.defenses.defense import (
    AlignmentDefense,
    AlignmentDefenseConfig,
)
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.utils.datasets import get_config_cache_path
from tamperbench.whitebox.utils.models.default_tokens import DEFAULT_PAD_TOKEN
from tamperbench.whitebox.utils.names import DefenseName
from tamperbench.whitebox.utils.ops import run_in_isolation
from tamperbench.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer


@dataclass
class DeepAlignConfig(AlignmentDefenseConfig):
    """Configuration for DeepAlign.

    Attributes:
        input_checkpoint_path: Model to align (inherited).
        output_checkpoint_path: Where the aligned model is saved (inherited).
        harmful_dataset_name: Dataset with ``prompt``/``response`` (harmful)/``refusal`` columns.
        harmful_dataset_split: Split of the harmful dataset.
        benign_dataset_name: Alpaca-style dataset for utility prompts.
        benign_dataset_split: Split of the benign dataset.
        num_harmful_samples: Number of harmful triples to use.
        num_benign_samples: Number of benign examples to mix in (0 disables). The default
            is 4x ``num_harmful_samples``, i.e. 20% harmful / 80% benign.
        distill_benign_responses: Answer the benign prompts with the input model and train
            on those answers, instead of the dataset's own answers.
        benign_max_new_tokens: Generation length cap for the distilled answers.
        max_prefix_tokens: Upper bound on k, the number of harmful tokens before the refusal.
        augment_prob: Probability a harmful example becomes a recovery example (else a plain refusal).
        learning_rate: Optimizer learning rate (full fine-tuning).
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
    num_harmful_samples: int = 256
    num_benign_samples: int = 1024  # 4x harmful -> 20/80 mix
    distill_benign_responses: bool = True
    benign_max_new_tokens: int = 512
    max_prefix_tokens: int = 100
    augment_prob: float = 0.5
    learning_rate: float = 2e-5
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 10  # effective batch 80 on one GPU
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

        print("[DeepAlign 1/5] Loading tokenizer and data...")
        tokenizer = _load_tokenizer(str(cfg.input_checkpoint_path))
        harmful_triples = load_harmful_triples(
            cfg.harmful_dataset_name, cfg.harmful_dataset_split, cfg.num_harmful_samples, cfg.seed
        )
        benign_pairs: list[dict[str, Any]] = load_benign_prompts(
            cfg.benign_dataset_name, cfg.benign_dataset_split, cfg.num_benign_samples, cfg.seed
        )
        print(f"   {len(harmful_triples)} harmful triples, {len(benign_pairs)} benign prompts")

        print("[DeepAlign 2/5] Answering benign prompts with the input model...")
        if cfg.distill_benign_responses:
            benign_pairs = self._distill_benign_responses(tokenizer, benign_pairs)
        else:
            print("   skipped (distill_benign_responses=False): using the dataset's answers")

        print("[DeepAlign 3/5] Tokenizing...")
        train_dataset = tokenize_examples(
            tokenizer,
            harmful_triples,
            benign_pairs,
            max_prefix_tokens=cfg.max_prefix_tokens,
            max_seq_length=cfg.max_seq_length,
        )
        print(f"   {len(train_dataset)} training examples")

        print(f"[DeepAlign 4/5] Full fine-tuning for {cfg.num_train_epochs} epochs...")
        model = _load_model(str(cfg.input_checkpoint_path))
        self._train(model, tokenizer, train_dataset)

        print("[DeepAlign 5/5] Cleaning up...")
        dealloc_model_and_tokenizer(model, tokenizer)

        print(f"DeepAlign done. Aligned model saved to {cfg.output_checkpoint_path}")
        return cfg.output_checkpoint_path

    def _distill_benign_responses(
        self, tokenizer: PreTrainedTokenizer, benign_pairs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Replace the dataset's answers with the input model's own (cached on disk).

        Generation runs in a subprocess so vLLM's GPU memory is fully released
        before training. Results are cached by model path and data settings; delete
        the cache file if the checkpoint at that path has changed.
        """
        cfg = self.defense_config
        if not benign_pairs:
            return benign_pairs

        cache_path = get_config_cache_path(
            "deep_align_benign_responses",
            {
                "model": str(cfg.input_checkpoint_path),
                "dataset": cfg.benign_dataset_name,
                "split": cfg.benign_dataset_split,
                "num_samples": cfg.num_benign_samples,
                "seed": cfg.seed,
                "max_new_tokens": cfg.benign_max_new_tokens,
            },
        )
        if cache_path.exists():
            print(f"   using cached responses from {cache_path}")
            cached = datasets.load_dataset("json", data_files=str(cache_path), split="train")
            return [cast(dict[str, Any], row) for row in cached]

        prompt_ids = [encode_prompt(tokenizer, pair["prompt"]) for pair in benign_pairs]
        generations = run_in_isolation(
            target=generate_responses_from_ids,
            args=(str(cfg.input_checkpoint_path), prompt_ids, cfg.benign_max_new_tokens),
            error_context="DeepAlign benign response generation",
        )
        distilled = [
            {"prompt": pair["prompt"], "response": text, "finished": finished}
            for pair, (text, finished) in zip(benign_pairs, generations, strict=True)
        ]
        num_cut_off = sum(not row["finished"] for row in distilled)
        print(f"   {len(distilled)} responses ({num_cut_off} hit the length cap; trained without EOS)")

        datasets.Dataset.from_list(distilled).to_json(cache_path, orient="records", lines=True)
        print(f"   cached to {cache_path}")
        return distilled

    def _train(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, train_dataset: datasets.Dataset) -> None:
        """Full fine-tuning with the causal-LM loss; the -100 labels already encode what to learn."""
        cfg = self.defense_config

        # If a pad token was added, the embedding matrix needs a row for it.
        model.resize_token_embeddings(len(tokenizer))

        training_args = TrainingArguments(
            output_dir=str(cfg.output_checkpoint_path),
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            num_train_epochs=cfg.num_train_epochs,
            lr_scheduler_type="linear",
            warmup_ratio=0.0,
            optim="adamw_torch",
            gradient_checkpointing=True,
            bf16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=cfg.seed,
            # Keep prompt_ids/harmful_ids/target_ids: the collator builds the inputs from them.
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=SafetyRecoveryCollator(
                tokenizer,
                max_prefix_tokens=cfg.max_prefix_tokens,
                augment_prob=cfg.augment_prob,
                max_seq_length=cfg.max_seq_length,
                seed=cfg.seed,
            ),
        )
        trainer.train()
        trainer.save_model(str(cfg.output_checkpoint_path))  # also saves the tokenizer
        trainer.accelerator.free_memory()
