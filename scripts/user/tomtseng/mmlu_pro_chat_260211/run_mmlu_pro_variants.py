"""Re-run an attack's best trial and evaluate with 4 MMLU-Pro prompt format variants.

For a given model-attack pair, this script:
1. Reads best.json to get the top trial's hyperparameters.
2. Re-runs the attack to produce a model checkpoint.
3. Runs 4 MMLU-Pro evaluations on the checkpoint:
   - few-shot + continuation (original)
   - zero-shot + continuation
   - few-shot + chat template
   - zero-shot + chat template
4. Deletes the checkpoint to free disk space.

Usage:
    uv run scripts/user/tomtseng/mmlu_pro_chat/run_mmlu_pro_variants.py \
        meta-llama/Llama-3.1-8B-Instruct \
        --model-alias llama3_8b_instruct \
        --attack lora_finetune \
        --source-results-dir results/nov7_trial \
        --output-results-dir results/mmlu_pro_chat
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any, TypedDict

import tamperbench.whitebox.attacks  # pyright: ignore[reportUnusedImport]  # ensure registry side-effects
import tamperbench.whitebox.evals  # noqa: F401  # pyright: ignore[reportUnusedImport]  # ensure registry side-effects
from tamperbench.utils import get_repo_root
from tamperbench.whitebox.attacks.registry import ATTACKS_REGISTRY
from tamperbench.whitebox.evals.mmlu_pro.mmlu_pro import (
    MMLUProEvaluationConfig,
    MMLUProTestEvaluation,
    MMLUProValEvaluation,
)
from tamperbench.whitebox.utils import AttackName, EvalPath
from tamperbench.whitebox.utils.benchmark.path_generation import get_trial_dir_name
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import ConfigKeys

REPO_ROOT = get_repo_root()


class _MMLUProVariant(TypedDict):
    name: str
    n_shots: int
    use_chat_template: bool


MMLU_PRO_VARIANTS: list[_MMLUProVariant] = [
    {"name": "fewshot_continuation", "n_shots": 5, "use_chat_template": False},
    {"name": "zeroshot_continuation", "n_shots": 0, "use_chat_template": False},
    {"name": "fewshot_chat", "n_shots": 5, "use_chat_template": True},
    {"name": "zeroshot_chat", "n_shots": 0, "use_chat_template": True},
]


def load_best_trial(source_results_dir: Path, model_alias: str, attack_name: str) -> dict[str, Any]:
    """Load the top trial info from best.json."""
    best_json_path = source_results_dir / model_alias / attack_name / "optuna_single" / "best.json"
    with open(best_json_path) as f:
        best = json.load(f)
    return best["top_trials"][0]


def load_best_config(source_results_dir: Path, model_alias: str, attack_name: str) -> dict[str, Any]:
    """Load the top trial's merged config from best.yaml (as parsed from the study DB)."""
    import yaml

    best_yaml_path = source_results_dir / model_alias / attack_name / "optuna_single" / "best.yaml"
    with open(best_yaml_path) as f:
        configs = yaml.safe_load(f)
    return configs["top_1"]


def run_attack_and_evaluate(
    pretrained_model_path: str,
    model_alias: str,
    attack_name_str: str,
    source_results_dir: Path,
    output_results_dir: Path,
    eval_split: str,
    random_seed: int = 42,
    chat_template_model: str | None = None,
    skip_attack: bool = False,
) -> None:
    """Re-run attack with best hyperparameters and evaluate with all MMLU-Pro variants."""
    # TODO: embedding_attack modifies embeddings at inference time and never saves a
    # checkpoint, so our standard evaluate-from-checkpoint flow doesn't apply.
    if attack_name_str == "embedding_attack":
        print(f"Skipping embedding_attack for {model_alias} (not supported)")
        return

    attack_name = AttackName(attack_name_str)

    # Load best trial config
    best_trial = load_best_trial(source_results_dir, model_alias, attack_name_str)
    config = load_best_config(source_results_dir, model_alias, attack_name_str)
    trial_number = best_trial["trial_number"]

    print(f"Model: {model_alias}, Attack: {attack_name_str}, Best trial: {trial_number}")
    print(f"Config: {json.dumps(config, indent=2, default=str)}")

    # Set up attack output directory
    attack_out_dir = output_results_dir / model_alias / attack_name_str / get_trial_dir_name(trial_number)

    # Check if all variant evaluations are already complete
    eval_name = "mmlu_pro_val" if eval_split == "val" else "mmlu_pro"
    all_done = all(
        (
            attack_out_dir
            / "mmlu_pro_variants"
            / v["name"]
            / EvalPath.CHECKPOINT_EVAL_DIR
            / eval_name
            / EvalPath.EVALUATION
        ).exists()
        for v in MMLU_PRO_VARIANTS
    )
    if all_done:
        print(f"All variants already evaluated for {model_alias}/{attack_name_str}, skipping.")
        return

    # Prepare attack config
    attack_config_cls, attack_cls = ATTACKS_REGISTRY[attack_name]
    attack_config_data = copy.deepcopy(config)
    attack_config_data.update(
        {
            "input_checkpoint_path": pretrained_model_path,
            "out_dir": str(attack_out_dir),
            ConfigKeys.RANDOM_SEED: random_seed,
        }
    )
    # Override evals to empty - we'll run evals ourselves
    attack_config_data[ConfigKeys.EVALS] = []

    attacker = attack_cls(attack_config=attack_config_cls.from_dict(data=attack_config_data))

    # Run attack (training only)
    checkpoint_path = Path(attacker.output_checkpoint_path)
    if skip_attack:
        if not checkpoint_path.exists():
            print(f"Checkpoint not found at {checkpoint_path}, skipping (--skip-attack)")
            return
        print(f"Using existing checkpoint at {checkpoint_path}")
    else:
        print(f"Running attack to produce checkpoint at {checkpoint_path}")
        if not checkpoint_path.exists():
            attacker.run_attack()
        print(f"Checkpoint ready at {checkpoint_path}")

    # Build ModelConfig from the attack config
    model_config_dict = config.get("model_config", {})
    model_config = ModelConfig.from_dict(model_config_dict)

    # Select evaluation class
    eval_cls = MMLUProValEvaluation if eval_split == "val" else MMLUProTestEvaluation

    # Run 4 MMLU-Pro evaluation variants
    for variant in MMLU_PRO_VARIANTS:
        variant_name = variant["name"]
        variant_out_dir = attack_out_dir / "mmlu_pro_variants" / variant_name

        print(f"\n--- Running MMLU-Pro variant: {variant_name} ---")
        print(f"  n_shots={variant['n_shots']}, use_chat_template={variant['use_chat_template']}")

        eval_config = MMLUProEvaluationConfig(
            model_checkpoint=attacker.output_checkpoint_path,
            out_dir=str(variant_out_dir),
            model_config=model_config,
            n_shots=variant["n_shots"],
            use_chat_template=variant["use_chat_template"],
            chat_template_model=chat_template_model if variant["use_chat_template"] else None,
        )

        evaluator = eval_cls(eval_config)
        results = evaluator.run_evaluation()
        print(f"  Results: {results.to_dicts()}")

    # Clean up checkpoint (but not if it's the pretrained model itself, e.g. no_weight_modification)
    checkpoint_path = Path(attacker.output_checkpoint_path)
    if checkpoint_path.exists() and str(checkpoint_path) != pretrained_model_path:
        print(f"\nDeleting checkpoint at {checkpoint_path}")
        shutil.rmtree(checkpoint_path)
    else:
        print("\nSkipping checkpoint cleanup (checkpoint is the pretrained model)")

    print(f"\nDone with {model_alias}/{attack_name_str}")


def main() -> None:
    """Run MMLU-Pro prompt format variants on a model-attack pair."""
    parser = argparse.ArgumentParser(description="Run MMLU-Pro prompt format variants on a model-attack pair")
    parser.add_argument("pretrained_model_path", type=str, help="HuggingFace model path or local checkpoint")
    parser.add_argument("--model-alias", type=str, required=True, help="Model alias (e.g., llama3_8b_instruct)")
    parser.add_argument("--attack", type=str, required=True, help="Attack name (e.g., lora_finetune)")
    parser.add_argument(
        "--source-results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "nov7_trial",
        help="Directory with existing results containing best.json",
    )
    parser.add_argument(
        "--output-results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mmlu_pro_chat",
        help="Directory to store new evaluation results",
    )
    parser.add_argument("--eval-split", choices=["val", "test"], default="val", help="MMLU-Pro split to evaluate on")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--chat-template-model",
        type=str,
        default=None,
        help="Model whose chat template to use when the primary tokenizer has no chat_template (e.g. for base models)",
    )
    parser.add_argument(
        "--skip-attack",
        action="store_true",
        # This was useful for clearing out checkpoints when I ran out of space
        # on the CAIS cluster---I ran this clear out all the interrupted jobs
        # that had saved checkpoints but crashed afterwards before deleting
        # their checkpoints
        help="Skip the attack step. If the checkpoint doesn't exist, exit gracefully instead of running the attack.",
    )

    args = parser.parse_args()

    run_attack_and_evaluate(
        pretrained_model_path=args.pretrained_model_path,
        model_alias=args.model_alias,
        attack_name_str=args.attack,
        source_results_dir=args.source_results_dir,
        output_results_dir=args.output_results_dir,
        eval_split=args.eval_split,
        random_seed=args.random_seed,
        chat_template_model=args.chat_template_model,
        skip_attack=args.skip_attack,
    )


if __name__ == "__main__":
    main()
