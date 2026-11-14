"""Re-run an attack's best trial to regression-test the chat-attack refactoring.

For a given model-attack pair, this script:
1. Reads best.yaml to get the top trial's merged config (hyperparameters + model_config).
2. Reads best.json to get the top trial's trial_number.
3. Re-runs the attack + standard evals (strong_reject, mmlu_pro_val) via attacker.benchmark().
4. Deletes the checkpoint to free disk space.

Usage:
    uv run scripts/user/tomtseng/chat_template_refactor_testing_260217/run_regression.py \
        Qwen/Qwen3-0.6B \
        --model-alias qwen3_0_6b \
        --attack lora_finetune \
        --source-results-dir /data/saad_hossain/SafeTuneBed/results/nov7_trial \
        --output-results-dir results/chat_template_refactor_testing_260217
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

import tamperbench.whitebox.attacks  # pyright: ignore[reportUnusedImport]  # ensure registry side-effects
import tamperbench.whitebox.evals  # noqa: F401  # pyright: ignore[reportUnusedImport]  # ensure registry side-effects
from tamperbench.utils import get_repo_root
from tamperbench.whitebox.attacks.registry import ATTACKS_REGISTRY
from tamperbench.whitebox.utils import AttackName
from tamperbench.whitebox.utils.benchmark.path_generation import get_trial_dir_name
from tamperbench.whitebox.utils.names import ConfigKeys, EvalPath, TemplateName

REPO_ROOT = get_repo_root()

# Legacy template names from nov7_trial that have been renamed in the refactor.
_LEGACY_TEMPLATE_NAMES: dict[str, str] = {
    "qwen": TemplateName.NATIVE,
    "llama3": TemplateName.NATIVE,
    "mistral": TemplateName.NATIVE,
}


def _normalize_template_name(config: dict[str, Any]) -> None:
    """Map legacy model-specific template names to their refactored equivalents."""
    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        return
    template = model_config.get("template")
    if template in _LEGACY_TEMPLATE_NAMES:
        model_config["template"] = _LEGACY_TEMPLATE_NAMES[template]


def load_best_trial(source_results_dir: Path, model_alias: str, attack_name: str) -> dict[str, Any]:
    """Load the top trial info from best.json."""
    best_json_path = source_results_dir / model_alias / attack_name / "optuna_single" / "best.json"
    with open(best_json_path) as f:
        best = json.load(f)
    return best["top_trials"][0]


def load_best_config(source_results_dir: Path, model_alias: str, attack_name: str) -> dict[str, Any]:
    """Load the top trial's merged config from best.yaml."""
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
    random_seed: int = 42,
) -> None:
    """Re-run attack with best hyperparameters and evaluate with standard evals."""
    if attack_name_str == "embedding_attack":
        print(f"Skipping embedding_attack for {model_alias} (not supported)")
        return

    attack_name = AttackName(attack_name_str)

    # Load best trial config
    best_trial = load_best_trial(source_results_dir, model_alias, attack_name_str)
    config = load_best_config(source_results_dir, model_alias, attack_name_str)
    _normalize_template_name(config)
    trial_number = best_trial["trial_number"]

    print(f"Model: {model_alias}, Attack: {attack_name_str}, Best trial: {trial_number}")
    print(f"Config: {json.dumps(config, indent=2, default=str)}")

    # Set up attack output directory
    attack_out_dir = output_results_dir / model_alias / attack_name_str / get_trial_dir_name(trial_number)

    # Check if evaluations are already complete
    eval_names = config.get("evals", [])
    eval_dir = attack_out_dir / EvalPath.CHECKPOINT_EVAL_DIR
    if eval_names and all((eval_dir / en / EvalPath.EVALUATION).exists() for en in eval_names):
        print(f"All evaluations already complete for {model_alias}/{attack_name_str}, skipping.")
        return

    # Prepare attack config -- use the evals from the loaded config
    attack_config_cls, attack_cls = ATTACKS_REGISTRY[attack_name]
    attack_config_data = copy.deepcopy(config)
    attack_config_data.update(
        {
            "input_checkpoint_path": pretrained_model_path,
            "out_dir": str(attack_out_dir),
            ConfigKeys.RANDOM_SEED: random_seed,
        }
    )

    attacker = attack_cls(attack_config=attack_config_cls.from_dict(data=attack_config_data))

    # Run attack + evaluations via benchmark()
    print(f"Running benchmark (attack + evals) for {model_alias}/{attack_name_str}...")
    results_df = attacker.benchmark()
    print(f"Results: {results_df.to_dicts()}")

    # Clean up checkpoint (but not for no_weight_modification which points to pretrained model)
    checkpoint_path = Path(attacker.output_checkpoint_path)
    if checkpoint_path.exists() and str(checkpoint_path) != pretrained_model_path:
        print(f"\nDeleting checkpoint at {checkpoint_path}")
        attacker.delete_output_checkpoint()
    else:
        print("\nSkipping checkpoint cleanup (checkpoint is the pretrained model)")

    print(f"\nDone with {model_alias}/{attack_name_str}")


def main() -> None:
    """Run chat-attack regression test for a single model-attack pair."""
    parser = argparse.ArgumentParser(description="Chat-attack regression test: re-run best trial with standard evals")
    parser.add_argument("pretrained_model_path", type=str, help="HuggingFace model path or local checkpoint")
    parser.add_argument("--model-alias", type=str, required=True, help="Model alias (e.g., qwen3_0_6b)")
    parser.add_argument("--attack", type=str, required=True, help="Attack name (e.g., lora_finetune)")
    parser.add_argument(
        "--source-results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "nov7_trial",
        help="Directory with existing results containing best.json/best.yaml",
    )
    parser.add_argument(
        "--output-results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "chat_template_refactor_testing_260217",
        help="Directory to store new regression test results",
    )
    parser.add_argument("--random-seed", type=int, default=42)

    args = parser.parse_args()

    run_attack_and_evaluate(
        pretrained_model_path=args.pretrained_model_path,
        model_alias=args.model_alias,
        attack_name_str=args.attack,
        source_results_dir=args.source_results_dir,
        output_results_dir=args.output_results_dir,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
