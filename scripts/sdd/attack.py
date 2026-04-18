#!/usr/bin/env python
"""Run MFT attack and/or evaluation on a hardened or vanilla model.

Replicates the LLM-finetune-Safety benchmark from the SDD paper (Table 1):
- Full-parameter fine-tune with k harmful samples from AdvBench (k = 10, 50, 100)
- Evaluates with PolicyEval (same GPT-4 judge as the paper), StrongREJECT,
  and MMLU-Pro

Use --num-harmful 0 to evaluate a model without any attack (the "Initial" row
in the paper's Table 1).

Usage:
    python scripts/sdd/attack.py --tier llama2_chat --num-harmful 100
    python scripts/sdd/attack.py --tier llama2_chat --num-harmful 100 --no-defense
    python scripts/sdd/attack.py --tier llama2_chat --num-harmful 0              # SDD initial
    python scripts/sdd/attack.py --tier llama2_chat --num-harmful 0 --no-defense # Vanilla initial

Expected runtime: ~30 min on A100 per k-shot setting (+ eval time).
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

from tamperbench.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetune,
    FullParameterFinetuneConfig,
)
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

    # Attack settings matching Qi et al. (2023) for Llama-2-7b-Chat
    parser.add_argument(
        "--num-harmful",
        type=int,
        default=100,
        help="Number of harmful samples (paper: 10, 50, 100; 0 for eval-only)",
    )
    parser.add_argument("--num-epochs", type=int, default=5, help="Fine-tuning epochs (Qi et al. 2023: 5)")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Attack learning rate (Qi et al. 2023: 5e-5)")
    parser.add_argument("--batch-size", type=int, default=10, help="Training batch size (Qi et al. 2023: 10)")

    args = parser.parse_args()
    load_dotenv()

    model = args.model or MODELS[args.tier]
    output_dir = get_output_dir(model)
    eval_only = args.num_harmful == 0

    # Determine input checkpoint and label
    defense_label = "vanilla" if args.no_defense else "sdd"
    if eval_only:
        input_checkpoint = model if args.no_defense else str(output_dir / "hardened")
        run_label = f"eval_{defense_label}_initial"
    else:
        input_checkpoint = model if args.no_defense else str(output_dir / "hardened")
        run_label = f"attack_{defense_label}_{args.num_harmful}shot"

    run_output = output_dir / run_label

    print("=" * 80)
    if eval_only:
        print(f"Evaluation (no attack) — {defense_label}")
    else:
        print(f"MFT Attack — {args.num_harmful}-shot, full-parameter, {defense_label}")
    print("=" * 80)
    print(f"Input model: {input_checkpoint}")
    print(f"Output: {run_output}")
    if not eval_only:
        print(f"Harmful samples: {args.num_harmful}")
        print(f"Epochs: {args.num_epochs}")
        print(f"LR: {args.learning_rate}")
    print("Template: llama2_chat")
    print("=" * 80)

    if not args.no_defense and not eval_only and not Path(input_checkpoint).exists():
        print(f"ERROR: Hardened model not found at {input_checkpoint}")
        print("Run harden.py first.")
        sys.exit(1)

    config = FullParameterFinetuneConfig(
        input_checkpoint_path=input_checkpoint,
        out_dir=str(run_output),
        evals=[EvalName.POLICY_EVAL, EvalName.STRONG_REJECT, EvalName.MMLU_PRO_VAL],
        # TODO: restore the Qi et al. (2023) system prompt once there's a
        # mechanism for threading a system prompt through the attack pipeline.
        # Previously this script packed Llama-2's [INST]/<<SYS>> markers and
        # the system prompt into user_prefix/assistant_prefix/end_turn with
        # template_name=PLAIN. Those ModelConfig fields were removed when
        # attacks were migrated to apply_chat_template, so we now fall back to
        # Llama-2's native chat template (no custom system prompt).
        model_config=ModelConfig(
            template_name=TemplateName.NATIVE,
            max_generation_length=1024,
            inference_batch_size=16,
        ),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=-1,
        lr_scheduler_type="cosine",  # Paper uses StepLR(gamma=0.85) per epoch, cosine is close
        optim="adamw_torch",
        dataset_size=args.num_harmful,
        poison_ratio=1.0,
        harmful_dataset="advbench",
        benign_dataset="bookcorpus",
        random_seed=42,
    )

    attack = FullParameterFinetune(attack_config=config)

    if eval_only:
        # Point evaluator at the input model directly
        attack.output_checkpoint_path = input_checkpoint
        results = attack.evaluate()
    else:
        results = attack.benchmark()
        # Clean up MFTed model checkpoint to save disk space
        attack.delete_output_checkpoint()
        print(f"Deleted MFTed model checkpoint at {attack.output_checkpoint_path}")

    print("\n" + "=" * 80)
    print(f"RESULTS — {run_label}")
    print("=" * 80)
    print(results)
    print("=" * 80)


if __name__ == "__main__":
    main()
