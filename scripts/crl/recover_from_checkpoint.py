#!/usr/bin/env python
"""Recover a merged model from a CRL training checkpoint.

This script loads a checkpoint .pt file, recreates the PEFT model structure,
loads the saved state dict, and merges it into a full model.

Usage:
    python scripts/crl/recover_from_checkpoint.py /path/to/checkpoint.pt /path/to/output
"""

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Must match the config used during training
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    """Recover a merged model from a CRL training checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", help="Path to checkpoint .pt file")
    parser.add_argument("output_path", help="Path to save merged model")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base model name")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    output_path = Path(args.output_path)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base model: {args.base_model}")
    print(f"Output: {output_path}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    print(f"   Checkpoint from step {checkpoint['step']}")

    # Determine the vocab size from checkpoint
    embed_key = "base_model.model.model.embed_tokens.weight"
    checkpoint_vocab_size = checkpoint["model_state_dict"][embed_key].shape[0]
    print(f"   Checkpoint vocab size: {checkpoint_vocab_size}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Keep on CPU to avoid OOM
    )
    # Resize to match checkpoint (may have added pad token during training)
    model.resize_token_embeddings(checkpoint_vocab_size)

    # Create PEFT model with same config
    print("\nCreating PEFT model...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    # Load state dict
    print("\nLoading checkpoint state dict...")
    peft_model.load_state_dict(checkpoint["model_state_dict"])

    # Merge and unload
    print("\nMerging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()  # pyright: ignore[reportCallIssue]

    # Save
    print(f"\nSaving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)  # pyright: ignore[reportCallIssue]
    tokenizer.save_pretrained(output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
