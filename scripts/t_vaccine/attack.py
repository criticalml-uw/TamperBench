#!/usr/bin/env python
"""Phase 2: Fine-tuning attack on an aligned (hardened) model.

Simulates the harmful fine-tuning scenario from the T-Vaccine paper (Section 5):
- Mix benign SST2 data with harmful BeaverTails data
- Default: 1000 samples with 10% harmful ratio (p=0.1)
- Fine-tune using LoRA for 20 epochs at lr=1e-5

The attack loads the alignment LoRA from Phase 1 and applies a second LoRA
on top via poisoned fine-tuning.

Usage:
    python scripts/t_vaccine/attack.py --tier llama2
    python scripts/t_vaccine/attack.py --tier llama2 --poison-ratio 0.1 --sample-num 1000

Expected runtime: ~1 hour on A100, ~23.5GB GPU memory.
Expected output: fine-tuned LoRA adapter (adapter_model.safetensors).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # pyright: ignore[reportImplicitRelativeImport]
    MODELS,
    get_output_dir,
)
from dotenv import load_dotenv

from tamperbench.whitebox.defenses.t_vaccine.train import UpstreamTrainingConfig, run_tar_training


def build_sst2_json(output_path: Path) -> Path:
    """Build SST2 benign dataset JSON from HuggingFace, matching original build_dataset.py."""
    if output_path.exists():
        print(f"SST2 dataset already exists at {output_path}, skipping build.")
        return output_path

    from datasets import load_dataset

    print("Building SST2 dataset from HuggingFace...")
    dataset = load_dataset("sst2")
    output_data: list[dict[str, Any]] = []
    for data in dataset["train"]:
        row: dict[str, Any] = data  # pyright: ignore[reportAssignmentType]
        item = {
            "instruction": "Analyze the sentiment of the input, and respond only positive or negative",
            "input": row["sentence"],
            "output": "negative" if row["label"] == 0 else "positive",
        }
        output_data.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"Built SST2 dataset: {len(output_data)} examples -> {output_path}")
    return output_path


def main():
    """Run fine-tuning attack on an aligned model."""
    parser = argparse.ArgumentParser(description="Run fine-tuning attack (Phase 2)")
    parser.add_argument("--tier", choices=MODELS.keys(), default="llama2")
    parser.add_argument("--model", type=str, help="Override base model path")
    parser.add_argument(
        "--no-defense", action="store_true", help="Attack the base model directly (skip alignment LoRA)"
    )

    # Attack hyperparameters (paper Section 5)
    parser.add_argument("--poison-ratio", type=float, default=0.1, help="Harmful data ratio (paper: 0.1)")
    parser.add_argument("--sample-num", type=int, default=1000, help="Total fine-tuning samples (paper: 1000)")
    parser.add_argument("--num-epochs", type=int, default=20, help="Fine-tuning epochs (paper: 20)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate (paper: 1e-5)")

    # T-Vaccine alignment hyperparameters (must match harden.py to find the right checkpoint)
    parser.add_argument("--rho", type=float, default=3.0, help="Perturbation intensity used in Phase 1")
    parser.add_argument("--lisa-activated-layers", type=int, default=8, help="Sampled layers used in Phase 1")
    parser.add_argument("--lisa-interval-steps", type=int, default=20, help="Layer resampling interval used in Phase 1")
    parser.add_argument(
        "--probability-steps", type=int, default=200, help="Probability recalc interval used in Phase 1"
    )
    parser.add_argument("--prompt-data-size", type=int, default=200, help="Harmful dataset size used in Phase 1")

    # Dataset paths
    parser.add_argument(
        "--alignment-dataset",
        type=str,
        default="anonymous4486/booster_dataset",
        help="HuggingFace dataset or local JSON for alignment data",
    )
    parser.add_argument(
        "--beaver-tails-dataset",
        type=str,
        default="PKU-Alignment/BeaverTails",
        help="BeaverTails dataset path",
    )

    args = parser.parse_args()
    load_dotenv()

    model = args.model or MODELS[args.tier]
    output_dir = get_output_dir(model)

    if args.no_defense:
        alignment_checkpoint = None
        attack_output = output_dir / "attack_no_defense"
    else:
        alignment_checkpoint = output_dir / "alignment"
        attack_output = output_dir / "attack"

    # Build SST2 benign dataset (same as original build_dataset.py)
    sst2_path = output_dir / "data" / "sst2.json"
    build_sst2_json(sst2_path)

    print("=" * 80)
    print(f"Fine-tuning Attack {'(no defense)' if args.no_defense else '(Phase 2)'}")
    print("=" * 80)
    print(f"Base model: {model}")
    print(f"Alignment LoRA: {alignment_checkpoint or '(none)'}")
    print(f"Output: {attack_output}")
    print(f"Poison ratio: {args.poison_ratio}")
    print(f"Sample num: {args.sample_num}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Benign dataset: {sst2_path}")
    print("=" * 80)

    if alignment_checkpoint is not None and not alignment_checkpoint.exists():
        print(f"ERROR: Alignment checkpoint not found at {alignment_checkpoint}")
        print("Run harden.py first to produce the alignment LoRA.")
        sys.exit(1)

    # The attack uses optimizer="normal" (standard fine-tuning, no perturbation).
    # It loads the alignment LoRA via lora_folder and trains a second LoRA on top.
    config = UpstreamTrainingConfig(
        model_name_or_path=model,
        output_dir=str(attack_output),
        data_path="PKU-Alignment/BeaverTails_dangerous",
        alignment_dataset_path=args.alignment_dataset,
        beaver_tails_dataset_path=args.beaver_tails_dataset,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100000,
        save_total_limit=0,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True,
        tf32=True,
        cache_dir="cache",
        optim="adamw_torch",
        optimizer="normal",  # Standard fine-tuning (no perturbation)
        rho=args.rho,
        lamb=5.0,
        alternating="",
        lora_folder=str(alignment_checkpoint) if alignment_checkpoint else "",
        lisa_activated_layers=args.lisa_activated_layers,
        lisa_interval_steps=args.lisa_interval_steps,
        prompt_data_size=args.prompt_data_size,
        probability_steps=args.probability_steps,
        system_evaluate=True,
        evaluate_step=False,
        max_length=200,
        poison_ratio=args.poison_ratio,
        sample_num=args.sample_num,
        benign_dataset=str(sst2_path),
        vaccine_ratio=0.0,
        guide_data_num=100,
        bad_sample_num=2000,
        harmful_dataset="BeaverTails",
        decoding_trust_dataset_path="",
        seed=42,
        log_dir=str(output_dir / "logs" / "attack"),
    )
    checkpoint = run_tar_training(config)

    print(f"\nAttack complete: {checkpoint}")


if __name__ == "__main__":
    main()
