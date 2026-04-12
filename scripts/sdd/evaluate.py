#!/usr/bin/env python
"""Evaluate a model without any attack (Initial/baseline scores).

Runs PolicyEval, StrongREJECT, and MMLU-Pro on the vanilla or SDD-hardened
model to get the "Initial" row in the paper's Table 1.

Usage:
    python scripts/sdd/evaluate.py --tier llama2_chat                # vanilla
    python scripts/sdd/evaluate.py --tier llama2_chat --sdd          # SDD-hardened

Expected runtime: ~30 min on A100.
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
    """Evaluate a model without any MFT attack."""
    parser = argparse.ArgumentParser(description="Evaluate model (no attack)")
    parser.add_argument("--tier", choices=MODELS.keys(), default="llama2_chat")
    parser.add_argument("--model", type=str, help="Override model path")
    parser.add_argument("--sdd", action="store_true", help="Evaluate SDD-hardened model instead of vanilla")

    args = parser.parse_args()
    load_dotenv()

    model = args.model or MODELS[args.tier]
    output_dir = get_output_dir(model)

    if args.sdd:
        input_checkpoint = str(output_dir / "hardened")
        eval_label = "eval_sdd_initial"
    else:
        input_checkpoint = model
        eval_label = "eval_vanilla_initial"

    eval_output = output_dir / eval_label

    print("=" * 80)
    print(f"Evaluation (no attack) — {'SDD' if args.sdd else 'Vanilla'}")
    print("=" * 80)
    print(f"Model: {input_checkpoint}")
    print(f"Output: {eval_output}")
    print("=" * 80)

    if args.sdd and not Path(input_checkpoint).exists():
        print(f"ERROR: Hardened model not found at {input_checkpoint}")
        print("Run harden.py first.")
        sys.exit(1)

    # Use a dummy attack config that points input and output at the same model.
    # benchmark() skips run_attack() when the output checkpoint already exists,
    # and delete_output_checkpoint() is a no-op when input == output.
    config = FullParameterFinetuneConfig(
        input_checkpoint_path=input_checkpoint,
        out_dir=str(eval_output),
        evals=[EvalName.POLICY_EVAL, EvalName.STRONG_REJECT, EvalName.MMLU_PRO_VAL],
        model_config=ModelConfig.from_dict(
            {
                "template": TemplateName.LLAMA2_CHAT,
                "max_generation_length": 1024,
                "inference_batch_size": 16,
            }
        ),
        per_device_train_batch_size=10,
        learning_rate=5e-5,
        num_train_epochs=1,
        max_steps=-1,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        dataset_size=0,
        poison_ratio=1.0,
        harmful_dataset="advbench",
        benign_dataset="bookcorpus",
        random_seed=42,
    )

    attack = FullParameterFinetune(attack_config=config)

    # The evaluator defaults to {out_dir}/tamperbench_model_checkpoint.
    # Since we're not running an attack, point it at the input model directly.
    attack.output_checkpoint_path = input_checkpoint

    results = attack.evaluate()

    print("\n" + "=" * 80)
    print(f"RESULTS — {eval_label}")
    print("=" * 80)
    print(results)
    print("=" * 80)


if __name__ == "__main__":
    main()
