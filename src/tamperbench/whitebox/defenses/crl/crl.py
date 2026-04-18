"""CRL (Contrastive Representation Learning) defense implementation.

Implements the defense from "Improving Large Language Model Safety with
Contrastive Representation Learning" (Simko et al., 2025).

The defense uses triplet loss to:
1. Keep benign representations similar to the original model
2. Push harmful representations away from their original positions
3. Separate benign from harmful representations
4. Cluster harmful representations together
"""

import gc
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from typing_extensions import override

from tamperbench.whitebox.defenses.crl.config import CRLConfig
from tamperbench.whitebox.defenses.crl.datasets import (
    prepare_benign_dataset,
    prepare_harmful_dataset,
)
from tamperbench.whitebox.defenses.crl.model import (
    create_frozen_model,
    create_trainable_model,
    load_tokenizer,
)
from tamperbench.whitebox.defenses.crl.trainer import run_training_loop
from tamperbench.whitebox.defenses.defense import AlignmentDefense
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.utils.names import DefenseName
from tamperbench.whitebox.utils.ops import run_in_isolation
from tamperbench.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer


@register_defense(DefenseName.CRL, CRLConfig)
class CRL(AlignmentDefense[CRLConfig]):
    """CRL defense: hardens models via contrastive representation learning.

    Implements triplet loss to separate benign and harmful representations
    while preserving model capabilities on benign inputs.
    """

    @override
    def run_defense(self) -> Path:
        """Run CRL defense to create hardened model.

        Returns:
            Path to the hardened model checkpoint.
        """
        return run_in_isolation(
            target=run_crl_training,
            args=(
                str(self.defense_config.input_checkpoint_path),
                str(self.defense_config.output_checkpoint_path),
                self.defense_config,
            ),
            error_context="CRL defense training",
        )


def _load_datasets(
    config: CRLConfig,
    tokenizer: PreTrainedTokenizer,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Load and prepare benign and harmful datasets for CRL training.

    Args:
        config: CRL configuration containing dataset names, splits, sizes, and
            training hyperparameters (batch_size, max_length, shuffle_seed).
        tokenizer: Tokenizer to use for encoding samples.

    Returns:
        Tuple of (benign_loader, harmful_loader), each a DataLoader yielding
        batches with keys ``input_ids``, ``attention_mask``, and ``response_mask``.
    """
    print(f"   Loading benign samples from {config.benign_dataset_name}...")
    benign_ds = prepare_benign_dataset(
        config.benign_dataset_name,
        config.benign_dataset_split,
        config.num_samples,
        tokenizer,
        config.max_length,
    )
    print(f"   Loaded {len(benign_ds)} benign samples")

    print(f"   Loading harmful samples from {config.harmful_dataset_name}...")
    harmful_ds = prepare_harmful_dataset(
        config.harmful_dataset_name,
        config.harmful_dataset_config,
        config.harmful_dataset_split,
        config.num_samples,
        tokenizer,
        config.max_length,
    )
    print(f"   Loaded {len(harmful_ds)} harmful samples")

    # Set format for PyTorch (include response_mask for token position selection)
    benign_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "response_mask"])
    harmful_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "response_mask"])

    # Use seeded generators for reproducible data ordering across runs/resumes
    benign_gen = torch.Generator().manual_seed(config.shuffle_seed)
    harmful_gen = torch.Generator().manual_seed(config.shuffle_seed)

    benign_loader: DataLoader[Any] = DataLoader(
        benign_ds,  # pyright: ignore[reportArgumentType]
        batch_size=config.batch_size,
        shuffle=True,
        generator=benign_gen,
    )
    harmful_loader: DataLoader[Any] = DataLoader(
        harmful_ds,  # pyright: ignore[reportArgumentType]
        batch_size=config.batch_size,
        shuffle=True,
        generator=harmful_gen,
    )

    return benign_loader, harmful_loader


def run_crl_training(model_path: str, output_path_str: str, config: CRLConfig) -> Path:
    """Run CRL defense training end-to-end.

    Loads models, prepares datasets, runs the triplet-loss training loop,
    merges LoRA weights, and saves the hardened model checkpoint. Designed
    to be called via ``run_in_isolation`` so that all GPU memory is released
    when the subprocess exits.

    Args:
        model_path: HuggingFace model name or path to the base model checkpoint.
        output_path_str: Directory path where the hardened model will be saved.
        config: CRL configuration with all defense hyperparameters.

    Returns:
        Path to the saved hardened model checkpoint directory.
    """
    output_path = Path(output_path_str)

    print("=" * 80)
    print("CRL Defense: Contrastive Representation Learning")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CRL defense requires CUDA. The training loop uses mixed-precision "
            "autocast with float16, which requires a CUDA device."
        )

    if (output_path / "config.json").exists():
        print(f"Skipping - model already exists at {output_path}")
        return output_path

    # Step 1: Load models and tokenizer
    print("\n[1/4] Loading models and tokenizer...")
    tokenizer = load_tokenizer(model_path)

    print("   Loading frozen reference model...")
    frozen_model = create_frozen_model(model_path, tokenizer)

    print("   Loading trainable model with LoRA...")
    trainable_model = create_trainable_model(model_path, tokenizer, config)

    # Step 2: Prepare datasets
    print("\n[2/4] Preparing datasets...")
    benign_loader, harmful_loader = _load_datasets(config, tokenizer)

    # Step 3: Training loop
    print("\n[3/4] Training with triplet loss...")
    run_training_loop(
        frozen_model,
        trainable_model,
        benign_loader,
        harmful_loader,
        config,
        output_path,
    )

    # Step 4: Save model
    print("\n[4/4] Saving hardened model...")
    output_path.mkdir(parents=True, exist_ok=True)

    # Merge LoRA weights and save
    merged_model = trainable_model.merge_and_unload()  # pyright: ignore[reportCallIssue]
    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Cleanup - use dealloc_model_and_tokenizer for proper GPU memory release
    dealloc_model_and_tokenizer(frozen_model, tokenizer)
    dealloc_model_and_tokenizer(merged_model, None)
    del trainable_model  # PeftModel wrapping the now-deallocated base model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("CRL Defense Complete! Hardened model saved to:")
    print(f"  {output_path}")
    print("=" * 80 + "\n")

    return output_path
