#!/usr/bin/env python
"""Harden a model using CRL defense.

Usage:
    python scripts/crl/harden.py --tier medium
    python scripts/crl/harden.py --model path/to/model
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root: python scripts/crl/harden.py
sys.path.insert(0, str(Path(__file__).parent))

from common import MODELS, get_output_dir  # pyright: ignore[reportImplicitRelativeImport]
from dotenv import load_dotenv

from tamperbench.whitebox.defenses.crl import CRL, CRLConfig


def run_harden(
    model_path: str,
    output_dir: Path,
    batch_size: int = 16,
    num_steps: int | None = None,
) -> str:
    """Harden a model using CRL defense.

    Returns path to the hardened model checkpoint.
    """
    print(f"\nHardening: {model_path}")
    print(f"Output: {output_dir}")

    config = CRLConfig(
        input_checkpoint_path=Path(model_path),
        output_checkpoint_path=output_dir / "hardened_model",
        batch_size=batch_size,
    )
    if num_steps is not None:
        config.num_steps = num_steps
    defense = CRL(defense_config=config)
    checkpoint = defense.run_defense()

    print(f"Done: {checkpoint}")
    return str(checkpoint)


def main():
    """Apply CRL hardening to a model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=MODELS.keys(), default="medium")
    parser.add_argument("--model", type=str, help="Override model path")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16 per paper)")
    parser.add_argument("--steps", type=int, help="Override number of training steps (default: 1100)")
    args = parser.parse_args()

    load_dotenv()

    model = args.model or MODELS[args.tier]
    output_dir = get_output_dir(model)

    run_harden(model, output_dir, args.batch_size, args.steps)


if __name__ == "__main__":
    main()
