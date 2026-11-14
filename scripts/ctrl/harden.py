#!/usr/bin/env python
"""Harden a model using CTRL defense.

Usage:
    python scripts/ctrl/harden.py --tier medium
    python scripts/ctrl/harden.py --model path/to/model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS, get_output_dir  # pyright: ignore[reportImplicitRelativeImport]
from dotenv import load_dotenv

from tamperbench.whitebox.defenses.ctrl import CTRL, CTRLConfig


def main() -> None:
    """Harden a model using CTRL defense."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=MODELS.keys(), default="medium")
    parser.add_argument("--model", type=str, help="Override model path")
    args = parser.parse_args()

    load_dotenv()

    model = args.model or MODELS[args.tier]
    output_dir = get_output_dir(model)

    print(f"Hardening: {model}")
    print(f"Output: {output_dir}")

    config = CTRLConfig(
        input_checkpoint_path=Path(model),
        output_checkpoint_path=output_dir / "hardened_model",
        clean_dataset_names=("tatsu-lab/alpaca", "databricks/databricks-dolly-15k", "PKU-Alignment/BeaverTails"),
    )
    defense = CTRL(defense_config=config)
    checkpoint = defense.run_defense()

    print(f"\nDone: {checkpoint}")


if __name__ == "__main__":
    main()
