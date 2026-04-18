#!/usr/bin/env python
"""Ablation: fine-tune on uncurated data with same CTRL hyperparams, then eval MMLU-Pro.

This isolates whether the MMLU-Pro drop is caused by CTRL's curation or by
the SFT procedure itself (full fine-tuning on short instruction data).

Uses the same dataset mix as CTRL (Alpaca + Dolly + BeaverTails) but without curation.

Results on Llama-3-8B-Instruct (baseline MMLU-Pro: 0.4393):
  - Alpaca-only uncurated SFT:  0.0643
  - Mixed 3-dataset uncurated SFT: 0.0571

Usage:
    python scripts/ctrl/ablation_uncurated.py --tier large
    python scripts/ctrl/ablation_uncurated.py --model meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import datasets
from config import MODELS, get_output_dir  # pyright: ignore[reportImplicitRelativeImport]
from dotenv import load_dotenv
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from tamperbench.whitebox.defenses.ctrl.ctrl import (
    CTRLConfig,
    load_clean_datasets,
    load_model,
    load_tokenizer,
    prepare_training_dataset,
)
from tamperbench.whitebox.evals.mmlu_pro.mmlu_pro import (
    MMLUProEvaluationConfig,
    MMLUProValEvaluation,
)
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.ops.isolation import run_in_isolation


def _run_training(base_model: str, output_path_str: str, num_samples: int) -> None:
    """Run fine-tuning in an isolated subprocess to avoid OOM with eval."""
    output_path = Path(output_path_str)
    config = CTRLConfig(
        input_checkpoint_path=Path(base_model),
        output_checkpoint_path=output_path,
        # Use the default 3-dataset mix (Alpaca + Dolly + BeaverTails)
        num_samples=num_samples,
    )

    print(f"\n[1/2] Loading clean datasets ({num_samples} samples)...")
    _, clean_samples_list = load_clean_datasets(config)
    clean_dataset = datasets.Dataset.from_list(clean_samples_list)

    # Use prepare_training_dataset with empty curated set → uses original responses
    empty_curated = datasets.Dataset.from_list([])
    training_dataset = prepare_training_dataset(clean_dataset, empty_curated)
    print(f"   Training dataset: {len(training_dataset)} samples (uncurated)")

    print("\n[2/2] Fine-tuning on uncurated data...")
    model = load_model(base_model)
    tokenizer = load_tokenizer(base_model)

    # Check if already done
    config_file = output_path / "config.json"
    model_file = output_path / "model.safetensors"
    if config_file.exists() and config_file.stat().st_size > 0 and model_file.exists():
        print(f"   Skipping training - model already exists at {output_path}")
        return

    model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    model.enable_input_require_grads()

    training_args = SFTConfig(
        output_dir=str(output_path),
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_checkpointing=True,
        optim=config.optim,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        completion_only_loss=True,
        max_length=512,
        bf16=True,
        report_to="none",
        ddp_backend=None,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=training_dataset,
        args=training_args,
    )

    # Resume from checkpoint if one exists
    checkpoint_dirs = list(output_path.glob("checkpoint-*"))
    resume_from = None
    if checkpoint_dirs:
        latest = max(checkpoint_dirs, key=lambda p: int(p.name.split("-")[1]))
        resume_from = str(latest)
        print(f"   Resuming from checkpoint: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(output_dir=str(output_path))
    print(f"   Saved to {output_path}")


def eval_mmlu_pro(model_checkpoint: str, out_dir: Path) -> float:
    """Run MMLU-Pro val evaluation and return accuracy."""
    eval_config = MMLUProEvaluationConfig(
        model_checkpoint=model_checkpoint,
        out_dir=str(out_dir),
        model_config=ModelConfig(
            user_prefix="",
            assistant_prefix="",
            end_turn="",
            max_generation_length=2048,
            inference_batch_size=8,
        ),
    )
    evaluation = MMLUProValEvaluation(eval_config=eval_config)
    results = evaluation.run_evaluation()
    accuracy = float(results["metric_value"][0])
    return accuracy


def main() -> None:
    """Run uncurated SFT ablation and MMLU-Pro evaluation."""
    parser = argparse.ArgumentParser(description="Ablation: uncurated SFT + MMLU-Pro eval")
    parser.add_argument("--tier", choices=MODELS.keys(), default="large")
    parser.add_argument("--model", type=str, help="Override model path")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only run eval")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval, only run training")
    args = parser.parse_args()

    load_dotenv()

    base_model = args.model or MODELS[args.tier]
    output_dir = get_output_dir(base_model)
    uncurated_path = output_dir / "ablation_uncurated"
    uncurated_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Ablation: Uncurated SFT -> MMLU-Pro")
    print(f"Model: {base_model}")
    print("=" * 80)

    # Fine-tune in isolated subprocess so GPU memory is fully freed before eval
    if not args.skip_train:
        model_path = str(uncurated_path / "model")
        run_in_isolation(
            target=_run_training,
            args=(base_model, model_path, 2000),
            error_context="Uncurated SFT fine-tuning",
        )

    # Eval MMLU-Pro on both baseline and fine-tuned
    if not args.skip_eval:
        print("\n" + "=" * 80)
        print("Evaluating MMLU-Pro...")
        print("=" * 80)

        print("\n[1/2] Baseline MMLU-Pro...")
        baseline_acc = eval_mmlu_pro(base_model, output_dir / "ablation_mmlu_baseline")
        print(f"   Baseline MMLU-Pro accuracy: {baseline_acc:.4f}")

        print("\n[2/2] Uncurated SFT MMLU-Pro...")
        sft_acc = eval_mmlu_pro(
            str(uncurated_path / "model"),
            output_dir / "ablation_mmlu_uncurated",
        )
        print(f"   Uncurated SFT MMLU-Pro accuracy: {sft_acc:.4f}")

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Baseline MMLU-Pro:       {baseline_acc:.4f}")
        print(f"Uncurated SFT MMLU-Pro:  {sft_acc:.4f}")
        print(f"Delta:                   {sft_acc - baseline_acc:+.4f}")

        if baseline_acc > 0:
            pct = (baseline_acc - sft_acc) / baseline_acc * 100
            print(f"Relative drop:           {pct:.1f}%")


if __name__ == "__main__":
    main()
