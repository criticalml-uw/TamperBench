#!/usr/bin/env python
"""Run LoRA attack on a model (hardened or baseline).

Usage:
    python scripts/booster/attack.py --model path/to/model --output-dir ./out
    python scripts/booster/attack.py --tier medium --hardened

The attack simulates the harmful fine-tuning scenario from the Booster paper:
- Mix benign fine-tuning data (SST2) with harmful data (BeaverTails)
- Default: 1000 samples with 10% harmful ratio
- Fine-tune using LoRA for 20 epochs at lr=1e-5 (paper Section 5.1)
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

from tamperbench.whitebox.attacks.lora_finetune.lora_finetune import (
    LoraFinetune,
    LoraFinetuneConfig,
)
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import EvalName


def main():
    """Run LoRA attack on a base or hardened model."""
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning attack on a model")
    parser.add_argument("--tier", choices=MODELS.keys(), default="medium")
    parser.add_argument("--model", type=str, help="Override model path")
    parser.add_argument("--hardened", action="store_true", help="Attack hardened model")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")

    # Attack hyperparameters (matching paper Section 5.1)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for fine-tuning (paper: 1e-5)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of fine-tuning epochs (paper: 20)",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size (paper: 10)")

    # Dataset configuration (matching paper Section 5.1)
    parser.add_argument(
        "--total-samples",
        type=int,
        default=1000,
        help="Total fine-tuning samples (paper: n=1000)",
    )
    parser.add_argument(
        "--harmful-ratio",
        type=float,
        default=0.1,
        help="Fraction of harmful samples (paper: p=0.1)",
    )
    parser.add_argument(
        "--benign-task",
        type=str,
        default="sst2",
        choices=["sst2", "agnews"],
        help="Benign fine-tuning task (paper: SST2)",
    )

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

    num_harmful = int(args.total_samples * args.harmful_ratio)
    num_benign = args.total_samples - num_harmful

    print("=" * 80)
    print("LoRA Fine-tuning Attack (Booster Paper Setup)")
    print("=" * 80)
    print(f"Attacking: {model}")
    print(f"Output: {output_dir}")
    print("-" * 40)
    print("Training parameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print("-" * 40)
    print("Dataset configuration:")
    print("WARNING: Dataset configuration parameters below are NOT YET IMPLEMENTED.")
    print("The attack currently uses the default LLM-LAT/harmful-dataset regardless of these settings.")
    print(f"  Total samples: {args.total_samples}")
    print(f"  Harmful ratio: {args.harmful_ratio} ({num_harmful} harmful)")
    print(f"  Benign task: {args.benign_task} ({num_benign} benign)")
    print("=" * 80)

    # Use LoRA config matching paper (Section 5.1):
    # "LoRA for efficient LLM training. The rank of the adaptor is set to 32"
    # TODO: LoraFinetuneConfig doesn't yet support total_samples, harmful_ratio,
    # or benign_task. The default dataset (LLM-LAT/harmful-dataset) is used instead
    # of the paper's SST2+BeaverTails mix. These params need to be added to the
    # attack config to match the paper's Section 5.1 setup.
    config = LoraFinetuneConfig(
        input_checkpoint_path=model,
        out_dir=str(output_dir),
        model_config=ModelConfig(
            # TODO: These prefixes should match the target model's chat template.
            # Using Llama-2 base (Alpaca) format as placeholder.
            user_prefix="### Instruction:\n",
            assistant_prefix="### Response:\n",
            end_turn="\n",
            max_generation_length=512,
            inference_batch_size=16,
        ),
        evals=[EvalName.STRONG_REJECT],
        random_seed=42,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=-1,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        lora_rank=32,  # Paper default (Section 5.1)
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    )
    attack = LoraFinetune(attack_config=config)
    results = attack.benchmark()

    print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
