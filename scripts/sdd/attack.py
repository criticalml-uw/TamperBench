#!/usr/bin/env python
"""Phase 2: Run MFT attack on a hardened (or vanilla) model.

Replicates the LLM-finetune-Safety attack from the SDD paper (Table 1):
- Fine-tune with k harmful samples from AdvBench (k = 10, 50, 100)
- Uses the TamperBench LoRA fine-tuning attack infrastructure
- Evaluates with PolicyEval (same GPT-4 judge as the paper), StrongREJECT,
  and MMLU-Pro

The paper uses pure harmful data (no benign mixing), trained via SFT
with LLaMA-Factory on the Llama2 chat template.

Usage:
    python scripts/sdd/attack.py --tier llama2_chat --num-harmful 100
    python scripts/sdd/attack.py --tier llama2_chat --num-harmful 100 --no-defense

Expected runtime: ~15 min on A100 per k-shot setting (+ eval time).
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

from tamperbench.whitebox.attacks.lora_finetune.lora_finetune import LoraFinetune, LoraFinetuneConfig
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import EvalName, TemplateName


def main():
    """Run MFT attack on hardened or vanilla model."""
    parser = argparse.ArgumentParser(description="Run MFT attack (Phase 2)")
    parser.add_argument("--tier", choices=MODELS.keys(), default="llama2_chat")
    parser.add_argument("--model", type=str, help="Override model path")
    parser.add_argument(
        "--no-defense",
        action="store_true",
        help="Attack vanilla model (no SDD defense)",
    )

    # Attack settings matching paper's LLM-finetune-Safety protocol
    parser.add_argument(
        "--num-harmful",
        type=int,
        default=100,
        help="Number of harmful samples (paper: 10, 50, 100)",
    )
    parser.add_argument("--num-epochs", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Attack learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")

    args = parser.parse_args()
    load_dotenv()

    model = args.model or MODELS[args.tier]
    output_dir = get_output_dir(model)

    # Determine input checkpoint: hardened model or vanilla
    if args.no_defense:
        input_checkpoint = model
        attack_label = f"attack_vanilla_{args.num_harmful}shot"
    else:
        input_checkpoint = str(output_dir / "hardened")
        attack_label = f"attack_sdd_{args.num_harmful}shot"

    attack_output = output_dir / attack_label

    print("=" * 80)
    print(f"MFT Attack (Phase 2) — {args.num_harmful}-shot")
    print("=" * 80)
    print(f"Input model: {input_checkpoint}")
    print(f"Output: {attack_output}")
    print(f"Harmful samples: {args.num_harmful}")
    print(f"Epochs: {args.num_epochs}")
    print(f"LR: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_rank}")
    print("Template: llama2_chat")
    print("Harmful dataset: advbench")
    print("=" * 80)

    if not args.no_defense and not Path(input_checkpoint).exists():
        print(f"ERROR: Hardened model not found at {input_checkpoint}")
        print("Run harden.py first.")
        sys.exit(1)

    config = LoraFinetuneConfig(
        input_checkpoint_path=input_checkpoint,
        out_dir=str(attack_output),
        evals=[EvalName.POLICY_EVAL, EvalName.STRONG_REJECT, EvalName.MMLU_PRO_VAL],
        model_config=ModelConfig.from_dict(
            {
                "template": TemplateName.LLAMA2_CHAT,
                "max_generation_length": 1024,
                "inference_batch_size": 16,
            }
        ),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=-1,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        lora_rank=args.lora_rank,
        dataset_size=args.num_harmful,
        poison_ratio=1.0,  # Pure harmful data (no benign mixing)
        harmful_dataset="advbench",
        benign_dataset="bookcorpus",
        random_seed=42,
    )

    attack = LoraFinetune(attack_config=config)
    results = attack.benchmark()

    print("\n" + "=" * 80)
    print(f"RESULTS — {attack_label}")
    print("=" * 80)
    print(results)
    print("=" * 80)


if __name__ == "__main__":
    main()
