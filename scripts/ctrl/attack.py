#!/usr/bin/env python
"""Run LoRA attack on a model.

Usage:
    python scripts/ctrl/attack.py --model path/to/model --output-dir ./out
    python scripts/ctrl/attack.py --tier medium --hardened
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS, get_output_dir  # pyright: ignore[reportImplicitRelativeImport]
from dotenv import load_dotenv

from tamperbench.whitebox.attacks.lora_finetune.lora_finetune import (
    LoraFinetune,
    LoraFinetuneConfig,
)
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import EvalName


def main() -> None:
    """Run LoRA attack on a base or hardened model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=MODELS.keys(), default="medium")
    parser.add_argument("--model", type=str, help="Override model path")
    parser.add_argument("--hardened", action="store_true", help="Attack hardened model")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    args = parser.parse_args()

    load_dotenv()

    base_model = args.model or MODELS[args.tier]
    base_output_dir = get_output_dir(base_model)

    if args.hardened:
        model = str(base_output_dir / "hardened_model")
        output_dir = args.output_dir or base_output_dir / "attack_hardened"
    else:
        model = base_model
        output_dir = args.output_dir or base_output_dir / "attack_baseline"

    print(f"Attacking: {model}")
    print(f"Output: {output_dir}")

    config = LoraFinetuneConfig(
        input_checkpoint_path=model,
        out_dir=str(output_dir),
        evals=[EvalName.STRONG_REJECT],
        model_config=ModelConfig(
            user_prefix="",
            assistant_prefix="",
            end_turn="",
            max_generation_length=512,
            inference_batch_size=8,
        ),
        random_seed=42,
        per_device_train_batch_size=8,
        learning_rate=1e-04,
        num_train_epochs=2,
        max_steps=-1,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        lora_rank=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
    )
    attack = LoraFinetune(attack_config=config)
    results = attack.benchmark()

    print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
