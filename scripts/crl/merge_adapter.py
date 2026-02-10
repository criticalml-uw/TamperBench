#!/usr/bin/env python
"""Merge LoRA adapter into base model for evaluation.

Usage:
    python scripts/crl/merge_adapter.py /path/to/adapter /path/to/merged_output
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Merge a LoRA adapter into its base model and save the result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_path", help="Path to LoRA adapter")
    parser.add_argument("output_path", help="Path to save merged model")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    output_path = Path(args.output_path)

    # Load adapter config to get base model
    config_path = adapter_path / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)

    base_model_name = config["base_model_name_or_path"]
    print(f"Base model: {base_model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_path}")

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Keep on CPU to avoid OOM
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load and merge adapter
    print("Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()  # pyright: ignore[reportCallIssue]

    # Save merged model
    print(f"Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Done!")


if __name__ == "__main__":
    main()
