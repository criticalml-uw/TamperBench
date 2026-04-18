"""Booster defense implementation.

Reference:
    Huang et al., "Booster: Tackling Harmful Fine-tuning for Large Language Models
    via Attenuating Harmful Perturbation", arXiv:2409.01586v4, 2025.
    https://github.com/git-disl/Booster

This implementation follows Algorithm 1 from the paper exactly.
All hyperparameters are set to match Table 9 and Section 5.1 of the paper.
"""

from pathlib import Path
from typing import cast

import datasets
import peft
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from typing_extensions import override

from tamperbench.whitebox.defenses.booster.config import BoosterConfig
from tamperbench.whitebox.defenses.booster.data import (
    DataCollatorForBooster,
    load_beavertails_data,
    prepare_dataset,
)
from tamperbench.whitebox.defenses.booster.trainer import BoosterTrainer
from tamperbench.whitebox.defenses.defense import AlignmentDefense
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.utils.models.default_tokens import DEFAULT_PAD_TOKEN
from tamperbench.whitebox.utils.names import DefenseName
from tamperbench.whitebox.utils.ops import run_in_isolation
from tamperbench.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer

# Re-export for backwards compatibility
__all__ = ["Booster", "BoosterConfig"]


@register_defense(DefenseName.BOOSTER, BoosterConfig)
class Booster(AlignmentDefense[BoosterConfig]):
    """Booster defense: hardens models via attenuating harmful perturbation.

    Implements the Booster defense from "Booster: Tackling Harmful Fine-tuning for
    Large Language Models via Attenuating Harmful Perturbation" (Huang et al., 2025).

    The defense works by:
    1. Loading alignment data (harmful prompt + safe answer pairs)
    2. Loading harmful data (harmful prompt + harmful answer pairs)
    3. Training with a regularizer that minimizes harmful loss reduction rate
       when the model is perturbed in the harmful gradient direction

    Key insight from paper (Section 3.2):
    "Harmful perturbation (taking a step over the gradient of the harmful data)
    contributes to reduction of harmful training/testing loss, eventually triggering
    the model to respond in a harmful way."

    The optimization objective (Equation 1):
    min_w f(w) + lambda * (h(w) - h(w - alpha * grad_h(w) / ||grad_h(w)||))

    where:
    - f(w) is the alignment loss
    - h(w) is the harmful loss
    - lambda is the regularizer intensity
    - alpha is the inner step size
    """

    @override
    def _run_defense(self) -> Path:
        """Run Booster defense to create hardened model.

        Returns:
            Path to the hardened model checkpoint.
        """
        return run_in_isolation(
            target=run_booster_training,
            args=(
                str(self.defense_config.input_checkpoint_path),
                str(self.defense_config.output_checkpoint_path),
                self.defense_config,
            ),
            error_context="Booster defense training",
        )


def _load_model(model_path: str) -> PreTrainedModel:
    """Load base model from checkpoint.

    Args:
        model_path: HuggingFace model name or path to the base model checkpoint.

    Returns:
        PreTrainedModel loaded in bfloat16 with automatic device mapping.
    """
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model


def _load_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """Load tokenizer from checkpoint with padding configured.

    Args:
        model_path: HuggingFace model name or path to the base model checkpoint.

    Returns:
        PreTrainedTokenizer with right-side padding and a pad token set.
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN})

    return tokenizer


def _apply_lora(model: PreTrainedModel, config: BoosterConfig) -> peft.PeftModel:
    """Apply LoRA adapters to model.

    Paper Section 5.1: "utilize LoRA for efficient LLM training.
    The rank of the adaptor is set to 32, and the LoRA's alpha is 4."

    Args:
        model: Base model to apply LoRA to.
        config: Booster configuration with LoRA hyperparameters
            (lora_r, lora_alpha, lora_target_modules).

    Returns:
        PeftModel with LoRA adapters applied and trainable parameters printed.
    """
    model.enable_input_require_grads()

    lora_config = peft.LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=0.0,  # Paper doesn't mention dropout
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
    )

    # get_peft_model returns PeftModel | PeftMixedModel, but with a single
    # LoRA config it's always PeftModel.
    peft_model = cast(peft.PeftModel, peft.get_peft_model(model, lora_config))
    peft_model.print_trainable_parameters()

    return peft_model


def _load_datasets(
    config: BoosterConfig,
    tokenizer: PreTrainedTokenizer,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Load and prepare alignment and harmful datasets.

    Paper Section 5.1 and Appendix B:
    "We use the alignment dataset and harmful dataset from (Rosati et al., 2024c),
    which is enriched from BeaverTails (Ji et al., 2023)."

    Args:
        config: Booster configuration with dataset names, sample counts,
            and max_seq_length.
        tokenizer: Tokenizer for encoding text into token IDs.

    Returns:
        Tuple of (alignment_dataset, harmful_dataset), each a HuggingFace
        Dataset ready for the BoosterTrainer.
    """
    # Load raw datasets
    raw_alignment = load_beavertails_data(
        config.alignment_dataset_name,
        config.num_alignment_samples,
    )
    raw_harmful = load_beavertails_data(
        config.harmful_dataset_name,
        config.num_harmful_samples,
    )

    # Prepare datasets
    alignment_dataset = prepare_dataset(
        raw_alignment,
        tokenizer,
        config.max_seq_length,
        is_harmful=False,  # Use safe responses
    )
    harmful_dataset = prepare_dataset(
        raw_harmful,
        tokenizer,
        config.max_seq_length,
        is_harmful=True,  # Use harmful responses
    )

    return alignment_dataset, harmful_dataset


def _train_model(
    model: PreTrainedModel | peft.PeftModel,
    tokenizer: PreTrainedTokenizer,
    alignment_dataset: datasets.Dataset,
    harmful_dataset: datasets.Dataset,
    config: BoosterConfig,
    output_path: Path,
) -> None:
    """Train model with Booster algorithm.

    Args:
        model: Model to train (with or without LoRA).
        tokenizer: Tokenizer for the data collator.
        alignment_dataset: Alignment data (harmful prompt + safe answer).
        harmful_dataset: Harmful data (harmful prompt + harmful answer).
        config: Booster configuration with training hyperparameters.
        output_path: Directory for saving checkpoints during training.
    """
    # Training arguments from paper (Appendix B)
    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        # gradient_checkpointing must be False for Booster
        # because we manually manipulate gradients across 3 backward passes
        gradient_checkpointing=False,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=20,  # Keep all checkpoints for evaluation
        bf16=True,
        remove_unused_columns=False,
        # IMPORTANT: Do NOT use gradient clipping - reference implementation doesn't
        # and it can interfere with the Booster regularizer
        max_grad_norm=0.0,
        # Disable DDP - model uses device_map="auto" for model parallelism
        ddp_backend=None,
    )

    # Data collator
    data_collator = DataCollatorForBooster(tokenizer=tokenizer)

    # Create Booster trainer
    trainer = BoosterTrainer(
        harmful_dataset=harmful_dataset,
        lambda_=config.lambda_,
        alpha=config.alpha,
        model=model,
        args=training_args,
        train_dataset=alignment_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Resume from checkpoint if one exists
    checkpoint_dirs = list(output_path.glob("checkpoint-*"))
    resume_from = None
    if checkpoint_dirs:
        # Find the latest checkpoint
        latest = max(checkpoint_dirs, key=lambda p: int(p.name.split("-")[1]))
        resume_from = str(latest)
        print(f"   Resuming from checkpoint: {resume_from}")

    # Train
    trainer.train(resume_from_checkpoint=resume_from)
    # Note: We don't call trainer.save_model() here because:
    # 1. For LoRA, it saves adapter files that conflict with the merged model
    # 2. The merged model is saved by run_booster_training() after training
    trainer.accelerator.free_memory()


def run_booster_training(model_path: str, output_path_str: str, config: BoosterConfig) -> Path:
    """Run Booster defense training end-to-end.

    Loads model and tokenizer, optionally applies LoRA, prepares datasets,
    trains with the Booster algorithm, merges LoRA weights, and saves the
    hardened model. Designed to be called via ``run_in_isolation`` so that
    all GPU memory is released when the subprocess exits.

    Args:
        model_path: HuggingFace model name or path to the base model checkpoint.
        output_path_str: Directory path where the hardened model will be saved.
        config: Booster configuration with all defense hyperparameters.

    Returns:
        Path to the saved hardened model checkpoint directory.
    """
    output_path = Path(output_path_str)

    # Check if already trained to avoid overwriting a good model
    if (output_path / "config.json").exists():
        print(f"Model already exists at {output_path}, skipping.")
        return output_path

    print("=" * 80)
    print("Booster Defense: Attenuating Harmful Perturbation")
    print("=" * 80)
    print(f"  Lambda (regularizer intensity): {config.lambda_}")
    print(f"  Alpha (inner step size): {config.alpha}")
    print(f"  Use LoRA: {config.use_lora}")
    if config.use_lora:
        print(f"  LoRA rank: {config.lora_r}")
        print(f"  LoRA alpha: {config.lora_alpha}")

    # Step 1: Load model and tokenizer
    print(f"\n[1/5] Loading model from {model_path}...")
    model = _load_model(model_path)
    tokenizer = _load_tokenizer(model_path)

    # Resize embeddings if needed
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    # Step 2: Apply LoRA if configured
    if config.use_lora:
        print(f"\n[2/5] Applying LoRA (r={config.lora_r}, alpha={config.lora_alpha})...")
        model = _apply_lora(model, config)
    else:
        print("\n[2/5] Skipping LoRA (full parameter training)...")
        model.enable_input_require_grads()

    # Step 3: Load datasets
    print(
        f"\n[3/5] Loading datasets ({config.num_alignment_samples} alignment, {config.num_harmful_samples} harmful)..."
    )
    alignment_dataset, harmful_dataset = _load_datasets(config, tokenizer)
    print(f"   Alignment samples: {len(alignment_dataset)}")
    print(f"   Harmful samples: {len(harmful_dataset)}")

    # Step 4: Train with Booster algorithm
    print(f"\n[4/5] Training with Booster algorithm ({config.num_train_epochs} epochs)...")
    _train_model(model, tokenizer, alignment_dataset, harmful_dataset, config, output_path)

    # Step 5: Save model
    print(f"\n[5/5] Saving hardened model to {output_path}...")

    if config.use_lora:
        # merge_and_unload is dispatched via __getattr__ on PeftModel,
        # so basedpyright cannot resolve its return type statically.
        merged = model.merge_and_unload()  # pyright: ignore[reportCallIssue]  # PeftModel.__getattr__ dispatch
        model = cast(PreTrainedModel, merged)

    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Clean up LoRA adapter files to avoid confusion when loading
    # (transformers tries to load adapters if these files exist)
    adapter_files = [
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
    ]
    for fname in adapter_files:
        fpath = output_path / fname
        if fpath.exists():
            fpath.unlink()
            print(f"   Removed leftover adapter file: {fname}")

    # Cleanup - model is always PreTrainedModel here (merged if LoRA was used)
    dealloc_model_and_tokenizer(cast(PreTrainedModel, model), tokenizer)

    print("\n" + "=" * 80)
    print("Booster Defense Complete! Hardened model saved to:")
    print(f"  {output_path}")
    print("=" * 80 + "\n")

    return output_path
