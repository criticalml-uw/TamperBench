#!/usr/bin/env python
"""Phase 1: Harden a model using T-Vaccine defense (alignment phase).

Produces LoRA adapter weights in <output_dir>/alignment/.

Usage:
    python scripts/t_vaccine/harden.py --tier llama2
    python scripts/t_vaccine/harden.py --tier llama2 --rho 3

Paper defaults (Section 5, Appendix):
    Model: meta-llama/Llama-2-7b-hf
    Epochs: 20, Batch size: 10, LR: 1e-3
    rho=3, S=8 layers, K=200 steps, N_h=200 harmful examples

Expected runtime: ~1 hour on A100, ~23.5GB GPU memory.
Expected output: full merged checkpoint (LoRA adapter is merged into the base
model before saving), loadable directly via AutoModelForCausalLM / vLLM.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # pyright: ignore[reportImplicitRelativeImport]
    MODELS,
    get_output_dir,
)
from dotenv import load_dotenv

from tamperbench.whitebox.defenses.t_vaccine.t_vaccine import TVaccineConfig, TVaccineDefense


def main():
    """Harden a model using T-Vaccine defense."""
    parser = argparse.ArgumentParser(description="Run T-Vaccine alignment (Phase 1)")
    parser.add_argument("--tier", choices=MODELS.keys(), default="llama2")
    parser.add_argument("--model", type=str, help="Override model path")

    # T-Vaccine hyperparameters (all paper defaults)
    parser.add_argument("--rho", type=float, default=3.0, help="Perturbation intensity (paper: 3)")
    parser.add_argument("--lisa-activated-layers", type=int, default=8, help="Sampled layers S (paper: 8)")
    parser.add_argument("--lisa-interval-steps", type=int, default=20, help="Layer resampling interval J (paper: 20)")
    parser.add_argument("--probability-steps", type=int, default=200, help="Probability recalc interval K (paper: 200)")
    parser.add_argument("--prompt-data-size", type=int, default=200, help="Harmful dataset size N_h (paper: 200)")
    parser.add_argument("--num-epochs", type=int, default=20, help="Training epochs (paper: 20)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate (paper: 1e-3, tuned on Llama-2-7B)")

    # Dataset paths: The original T-Vaccine paper uses
    # beavertails_with_refusals_train.json from rosati2024immunization, generated
    # via scripts/generate_paired_refusals.sh in
    # https://github.com/domenicrosati/representation-noising.
    # anonymous4486/booster_dataset appears to be a HuggingFace upload of that
    # same file (its underlying filename is beavertails_with_refusals_train.json).
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
    alignment_output = output_dir / "alignment"

    print("=" * 80)
    print("T-Vaccine Alignment (Phase 1)")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Output: {alignment_output}")
    print(f"rho: {args.rho}")
    print(f"Sampled layers (S): {args.lisa_activated_layers}")
    print(f"Layer resampling interval (J): {args.lisa_interval_steps}")
    print(f"Probability recalc interval (K): {args.probability_steps}")
    print(f"Harmful dataset size (N_h): {args.prompt_data_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 80)

    config = TVaccineConfig(
        input_checkpoint_path=Path(model),
        output_checkpoint_path=alignment_output,
        data_path="PKU-Alignment/BeaverTails_safe_alignment",
        alignment_dataset_path=args.alignment_dataset,
        beaver_tails_dataset_path=args.beaver_tails_dataset,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=1,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=100000,
        save_total_limit=0,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        tf32=True,
        cache_dir="cache",
        optim="adamw_torch",
        optimizer="mesfa",
        rho=args.rho,
        lamb=5.0,
        alternating="",
        lisa_activated_layers=args.lisa_activated_layers,
        lisa_interval_steps=args.lisa_interval_steps,
        prompt_data_size=args.prompt_data_size,
        probability_steps=args.probability_steps,
        system_evaluate=True,
        evaluate_step=False,
        max_length=200,
        sample_num=1000,
        guide_data_num=100,
        bad_sample_num=2000,
        harmful_dataset="BeaverTails",
        decoding_trust_dataset_path="",
        seed=42,
        log_dir=str(output_dir / "logs" / "alignment"),
    )
    defense = TVaccineDefense(defense_config=config)
    checkpoint = defense.run_defense()

    print(f"\nAlignment complete: {checkpoint}")


if __name__ == "__main__":
    main()
