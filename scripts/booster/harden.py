#!/usr/bin/env python
"""Harden a model using Booster defense.

Usage:
    python scripts/booster/harden.py --tier medium
    python scripts/booster/harden.py --model path/to/model
    python scripts/booster/harden.py --tier llama2 --lambda 5 --alpha 0.1

All hyperparameters default to paper values (Section 5.1, Appendix B).
"""

import argparse
import sys
from pathlib import Path

# Add script directory to sys.path so config.py is importable as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # pyright: ignore[reportImplicitRelativeImport]  # standalone script; sys.path set above
    MODELS,
    get_output_dir,
)
from dotenv import load_dotenv

from tamperbench.whitebox.defenses.booster import Booster, BoosterConfig


def main():
    """Harden a model using Booster defense."""
    parser = argparse.ArgumentParser(description="Harden a model using Booster defense")
    parser.add_argument("--tier", choices=MODELS.keys(), default="medium")
    parser.add_argument("--model", type=str, help="Override model path")

    # Booster hyperparameters (all paper defaults)
    parser.add_argument("--lambda", type=float, default=5.0, dest="lambda_", help="Regularizer intensity (paper: 5)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Inner step size (paper: 0.1)")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs (paper: 20)",
    )
    parser.add_argument(
        "--num-alignment-samples",
        type=int,
        default=5000,
        help="Number of alignment samples (paper: 5000)",
    )
    parser.add_argument(
        "--num-harmful-samples",
        type=int,
        default=5000,
        help="Number of harmful samples (paper: 5000)",
    )

    # LoRA settings (paper defaults)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (paper uses LoRA)")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (paper: 32)")
    parser.add_argument("--lora-alpha", type=int, default=4, help="LoRA alpha (paper: 4)")

    args = parser.parse_args()

    load_dotenv()

    model = args.model or MODELS[args.tier]
    output_dir = get_output_dir(model)

    print("=" * 80)
    print("Booster Defense Configuration")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Output: {output_dir}")
    print(f"Lambda: {args.lambda_}")
    print(f"Alpha: {args.alpha}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Alignment samples: {args.num_alignment_samples}")
    print(f"Harmful samples: {args.num_harmful_samples}")
    print(f"Use LoRA: {not args.no_lora}")
    if not args.no_lora:
        print(f"  LoRA rank: {args.lora_r}")
        print(f"  LoRA alpha: {args.lora_alpha}")
    print("=" * 80)

    config = BoosterConfig(
        input_checkpoint_path=Path(model),
        output_checkpoint_path=output_dir / "hardened_model",
        # Booster hyperparameters
        lambda_=args.lambda_,
        alpha=args.alpha,
        num_train_epochs=args.num_epochs,
        num_alignment_samples=args.num_alignment_samples,
        num_harmful_samples=args.num_harmful_samples,
        # LoRA settings
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    defense = Booster(defense_config=config)
    checkpoint = defense.run_defense()

    print(f"\nDone: {checkpoint}")


if __name__ == "__main__":
    main()
