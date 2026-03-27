#!/usr/bin/env python
"""Run full CTRL evaluation: harden, attack both, compare.

Usage:
    python scripts/ctrl/run_all.py --tier medium
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS, get_output_dir  # pyright: ignore[reportImplicitRelativeImport]
from dotenv import load_dotenv

from tamperbench.whitebox.attacks.lora_finetune.lora_finetune import (
    LoraFinetune,
    LoraFinetuneConfig,
)
from tamperbench.whitebox.defenses.ctrl import CTRL, CTRLConfig
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import EvalName, MetricName


def run_attack(model: str, output_dir: Path) -> float:
    """Run LoRA attack, return StrongREJECT score."""
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
    metrics = results.rows_by_key(key=EvaluationSchema.metric_name, unique=True)
    return metrics[MetricName.STRONG_REJECT_SCORE][0]


def main() -> None:
    """Run full CTRL evaluation pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=MODELS.keys(), default="medium")
    parser.add_argument("--model", type=str, help="Override model path")
    args = parser.parse_args()

    load_dotenv()

    base_model = args.model or MODELS[args.tier]
    output_dir = get_output_dir(base_model)

    print("=" * 80)
    print(f"CTRL Evaluation: {base_model}")
    print("=" * 80)

    # 1. Harden
    print("\n[1/3] Hardening model...")
    config = CTRLConfig(
        input_checkpoint_path=Path(base_model),
        output_checkpoint_path=output_dir / "hardened_model",
        clean_dataset_name="tatsu-lab/alpaca",
    )
    defense = CTRL(defense_config=config)
    hardened = str(defense.run_defense())

    # 2. Attack both
    print("\n[2/3] Attacking hardened model...")
    hardened_score = run_attack(hardened, output_dir / "attack_hardened")

    print("\n[3/3] Attacking baseline model...")
    baseline_score = run_attack(base_model, output_dir / "attack_baseline")

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Baseline StrongREJECT: {baseline_score:.4f}")
    print(f"Hardened StrongREJECT: {hardened_score:.4f}")
    print(f"Improvement: {baseline_score - hardened_score:+.4f}")


if __name__ == "__main__":
    main()
