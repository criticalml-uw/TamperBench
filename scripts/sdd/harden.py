#!/usr/bin/env python
"""Phase 1: Apply SDD defense to harden a model.

Trains the model on <harmful prompt, irrelevant benign response> pairs
constructed from BeaverTails + Alpaca/LIMA with cosine similarity filtering.

Paper defaults (Appendix):
    Model: meta-llama/Llama-2-7b-chat-hf
    LR: 5e-7, Steps: 500, Batch size: 24
    Dataset: 8K pairs from BeaverTails x (Alpaca + LIMA)
    Similarity threshold: 0.25 (SentenceBERT cosine)

Usage:
    python scripts/sdd/harden.py --tier llama2_chat
    python scripts/sdd/harden.py --tier minimal  # quick sanity check

Expected runtime: ~2 hours on A100.
Expected output: full model checkpoint.
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

from tamperbench.whitebox.defenses.sdd.sdd import SDD, SDDConfig


def main():
    """Apply SDD defense to create a hardened model."""
    parser = argparse.ArgumentParser(description="Run SDD defense (Phase 1)")
    parser.add_argument("--tier", choices=MODELS.keys(), default="llama2_chat")
    parser.add_argument("--model", type=str, help="Override model path")

    # SDD hyperparameters (all paper defaults)
    parser.add_argument("--num-samples", type=int, default=8000, help="Training pairs (paper: 8000)")
    parser.add_argument("--learning-rate", type=float, default=5e-7, help="Learning rate (paper: 5e-7)")
    parser.add_argument("--num-train-steps", type=int, default=500, help="Training steps (paper: 500)")
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size (paper: 24)")
    parser.add_argument("--similarity-threshold", type=float, default=0.25, help="Cosine sim threshold (paper: 0.25)")
    parser.add_argument("--use-reject-prefix", action="store_true", help="Use SDD_reject variant")

    args = parser.parse_args()
    load_dotenv()

    model = args.model or MODELS[args.tier]
    output_dir = get_output_dir(model)
    hardened_output = output_dir / "hardened"

    print("=" * 80)
    print("SDD Defense (Phase 1)")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Output: {hardened_output}")
    print(f"Samples: {args.num_samples}")
    print(f"LR: {args.learning_rate}")
    print(f"Steps: {args.num_train_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Variant: {'SDD_reject' if args.use_reject_prefix else 'SDD'}")
    print("=" * 80)

    config = SDDConfig(
        input_checkpoint_path=Path(model),
        output_checkpoint_path=hardened_output,
        num_samples=args.num_samples,
        learning_rate=args.learning_rate,
        num_train_steps=args.num_train_steps,
        per_device_train_batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold,
        use_reject_prefix=args.use_reject_prefix,
    )

    defense = SDD(defense_config=config)
    checkpoint = defense.run_defense()

    print(f"\nSDD defense complete: {checkpoint}")


if __name__ == "__main__":
    main()
