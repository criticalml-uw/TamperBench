"""Replicate Table 4 from Zhao et al. (2025) — RSN-Tune safety robustness.

Table 4 evaluates whether RSN-Tune preserves model safety when the model is
subsequently fine-tuned on a benign downstream task (GSM8K). The four conditions:

  Before     — base instruction-tuned model, no fine-tuning
  Undefended — fine-tuned on GSM8K (no defense)
  SN-Tune    — SN-Tune defense, then fine-tuned on GSM8K
  RSN-Tune   — RSN-Tune defense, then fine-tuned on GSM8K

Discrepancies vs. paper
-----------------------
1. **Safety metric.** The paper's "harmful score" is the average Attack Success
   Rate (ASR) across four adversarial methods — Direct Attack, GCG, AutoDAN,
   and PAIR — evaluated on the AdvBench harmful-behavior dataset (Zou et al.,
   2023) via the HarmBench framework. We use StrongREJECT (direct prompting
   only, no adversarial suffixes), so absolute numbers are not comparable.
2. **GSM8K fine-tuning hyperparameters.** The paper does not specify learning
   rate, batch size, or optimizer for the GSM8K fine-tuning step; only the
   dataset and number of epochs (1) are stated. We use lr=2e-5, batch_size=8,
   AdamW, full-parameter fine-tuning on the full GSM8K train set (7,473
   samples). Results will be sensitive to these choices.
3. **Capability metric.** The paper reports GSM8K accuracy. We report MMLU-Pro
   (val) instead, since TamperBench does not have a GSM8K eval. Both measure
   general reasoning capability, but absolute numbers differ.
4. **Defense learning rate.** The paper text says lr=1e-6 (Section 3), but the
   released code uses lr=2e-6 (train_neuron.py). We follow the paper (1e-6).

Usage:
    python scripts/rsn_tune/table4.py meta-llama/Llama-2-7b-chat-hf

    # With a results directory:
    python scripts/rsn_tune/table4.py meta-llama/Llama-2-7b-chat-hf \
        --results-dir results/rsn_tune_table4
"""

# pyright: reportUnusedCallResult=false, reportAny=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

# Import to trigger registry side-effects
import tamperbench.whitebox.attacks  # pyright: ignore[reportUnusedImport]
import tamperbench.whitebox.defenses  # pyright: ignore[reportUnusedImport]
import tamperbench.whitebox.evals  # noqa: F401  # pyright: ignore[reportUnusedImport]
from tamperbench.utils import get_repo_root
from tamperbench.whitebox.utils.benchmark.defense_config import (
    DefenseGridConfig,
)
from tamperbench.whitebox.utils.benchmark.defense_trial_manager import DefenseSweepTrialManager
from tamperbench.whitebox.utils.benchmark.io import yaml_to_dict
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import ConfigPath, DefenseName, EvalName

logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
DEFENSE_CONFIG_DIR = REPO_ROOT / "configs" / "whitebox" / "defenses"
ATTACK_CONFIG_DIR = REPO_ROOT / "configs" / "whitebox" / "attacks"


def evaluate_baseline(
    pretrained_model_path: str,
    eval_names: list[EvalName],
    model_config: ModelConfig,
    out_dir: Path,
) -> dict[str, float]:
    """Evaluate the base model (no defense, no attack) — the 'Before' condition."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return DefenseSweepTrialManager.evaluate_checkpoint(
        checkpoint_path=pretrained_model_path,
        eval_names=eval_names,
        model_config=model_config,
        out_dir=out_dir,
    )


def run_attack_only(
    pretrained_model_path: str,
    grid_config: DefenseGridConfig,
    model_config_dict: dict[str, object],
    out_dir: Path,
    random_seed: int,
    keep_checkpoints: bool = False,
) -> dict[str, float]:
    """Fine-tune the base model on GSM8K without any defense — the 'Original' condition.

    Runs the same attack configs as the defense trials, but directly on the
    pretrained model.
    """
    import copy

    from tamperbench.whitebox.evals.registry import EVALS_REGISTRY
    from tamperbench.whitebox.utils.benchmark.runners import run_attack_grid_configs

    eval_names = grid_config.post_attack_evals
    model_config = ModelConfig.from_dict(dict(model_config_dict))
    all_metrics: dict[str, float] = {}

    for attack_spec in grid_config.attacks:
        attack_out_dir = out_dir / str(attack_spec.name)
        attack_out_dir.mkdir(parents=True, exist_ok=True)

        grid_path = ATTACK_CONFIG_DIR / str(attack_spec.name) / ConfigPath.GRID_YAML
        raw_grid = yaml_to_dict(grid_path)

        config_grid: dict[str, dict[str, object]] = {}
        if attack_spec.config_name is not None:
            raw_value = raw_grid[attack_spec.config_name]
            if isinstance(raw_value, dict):
                config_grid[attack_spec.config_name] = raw_value
        else:
            for key, value in raw_grid.items():
                if isinstance(value, dict):
                    config_grid[key] = value

        for config_dict in config_grid.values():
            config_dict["model_config"] = copy.deepcopy(model_config_dict)
            config_dict["evals"] = [str(e) for e in eval_names]

        results = run_attack_grid_configs(
            attack_name=attack_spec.name,
            config_grid=config_grid,
            pretrained_model_path=pretrained_model_path,
            output_base_dir=attack_out_dir,
            random_seed=random_seed,
            eval_names=eval_names,
            cleanup_checkpoints=not keep_checkpoints,
        )

        for config_name, results_df in results.items():
            for eval_name in eval_names:
                eval_cls = EVALS_REGISTRY[eval_name]
                all_metrics[str(eval_name)] = float(eval_cls.load_result_objective(results_df))

    return all_metrics


def run_defense_condition(
    pretrained_model_path: str,
    defense_name: DefenseName,
    config_name: str,
    results_dir: Path,
    random_seed: int,
    model_alias: str,
    keep_checkpoints: bool = False,
    attack_config_override: str | None = None,
) -> dict[str, float]:
    """Run a defense + attack condition (SN-Tune or RSN-Tune)."""
    grid_path = DEFENSE_CONFIG_DIR / str(defense_name) / ConfigPath.GRID_YAML
    defense_grid = yaml_to_dict(grid_path)

    if config_name not in defense_grid:
        raise ValueError(f"Config '{config_name}' not found. Available: {list(defense_grid.keys())}")

    config_dict = defense_grid[config_name]
    if not isinstance(config_dict, dict):
        raise TypeError(f"Expected dict for config '{config_name}', got {type(config_dict)}")

    # Override the attack config name if specified
    if attack_config_override is not None:
        attacks = config_dict.get("attacks")
        if isinstance(attacks, list):
            for attack in attacks:
                if isinstance(attack, dict):
                    attack["config_name"] = attack_config_override

    grid_config = DefenseGridConfig.from_dict(config_dict)

    return DefenseSweepTrialManager.run_trial(
        defense_name=defense_name,
        defense_config_dict=config_dict,
        defense_eval_names=grid_config.defense_evals,
        post_attack_eval_names=grid_config.post_attack_evals,
        model_config_dict=grid_config.inference_model_config,
        attacks=grid_config.attacks,
        pretrained_model_path=pretrained_model_path,
        defense_results_dir=results_dir / model_alias / str(defense_name) / config_name,
        trial_number=0,
        random_seed=random_seed,
        attack_configs_dir=ATTACK_CONFIG_DIR,
        model_alias=model_alias,
        cleanup_checkpoints=not keep_checkpoints,
    )


def print_table(
    model_name: str,
    all_results: dict[str, dict[str, float]],
) -> None:
    """Print results in a format comparable to Table 4."""
    sr_key = str(EvalName.STRONG_REJECT)
    mmlu_key = str(EvalName.MMLU_PRO_VAL)
    pa = "post_attack.benign_full_parameter_finetune"

    # Build columns dynamically based on available results
    columns: list[tuple[str, dict[str, float], bool]] = []  # (label, data, is_defense)
    for key, label, is_def in [
        ("before", "Before", False),
        ("undefended", "Undefended", False),
        ("sn_tune", "SN-Tune", True),
        ("rsn_tune", "RSN-Tune", True),
        ("sn_tune_orig_code", "SN-OCode", True),
        ("rsn_tune_orig_code", "RSN-OCode", True),
    ]:
        if key in all_results:
            columns.append((label, all_results[key], is_def))

    col_w = 10
    header = f"{'Metric':<25}" + "".join(f"{label:>{col_w}}" for label, _, _ in columns)
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f"Table 4 Replication — {model_name}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    # StrongREJECT row
    def _get_sr(data: dict[str, float], is_defense: bool) -> float:
        if is_defense:
            return data.get(f"{pa}.{sr_key}", float("nan"))
        return data.get(sr_key, float("nan"))

    vals = "".join(f"{_get_sr(d, is_def):>{col_w}.3f}" for _, d, is_def in columns)
    print(f"{'StrongREJECT (harmful)':<25}{vals}")

    # Post-defense-only row
    def _get_sr_def(data: dict[str, float], is_defense: bool) -> str:
        if not is_defense:
            return f"{'—':>{col_w}}"
        v = data.get(f"defense.{sr_key}", float("nan"))
        return f"{v:>{col_w}.3f}"

    vals = "".join(_get_sr_def(d, is_def) for _, d, is_def in columns)
    print(f"{'  (post-defense only)':<25}{vals}")

    # MMLU-Pro row
    def _get_mmlu(data: dict[str, float], is_defense: bool) -> float:
        if is_defense:
            return data.get(f"{pa}.{mmlu_key}", float("nan"))
        return data.get(mmlu_key, float("nan"))

    vals = "".join(f"{_get_mmlu(d, is_def):>{col_w}.3f}" for _, d, is_def in columns)
    print(f"{'MMLU-Pro (capability)':<25}{vals}")

    def _get_mmlu_def(data: dict[str, float], is_defense: bool) -> str:
        if not is_defense:
            return f"{'—':>{col_w}}"
        v = data.get(f"defense.{mmlu_key}", float("nan"))
        return f"{v:>{col_w}.3f}"

    vals = "".join(_get_mmlu_def(d, is_def) for _, d, is_def in columns)
    print(f"{'  (post-defense only)':<25}{vals}")

    print(f"{'=' * len(header)}")
    print()
    print("Paper Table 4 reference (harmful score = avg ASR, lower = safer):")
    print("  Llama2-7B-Chat:         Before=0.0  Original=41.0  SN-Tune=38.0  RSN-Tune=26.0")
    print("  Mistral-7B-Instruct:    Before=36.0 Original=79.0  SN-Tune=72.0  RSN-Tune=41.0")
    print()
    print("Absolute values are NOT directly comparable (see docstring for discrepancies).")
    print("The expected qualitative pattern is: RSN-Tune < SN-Tune < Undefended.")


def main() -> None:
    """Run the Table 4 replication experiment."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Replicate Table 4 from Zhao et al. (2025) — RSN-Tune safety robustness.",
    )
    parser.add_argument(
        "pretrained_model_path",
        type=str,
        help="HuggingFace model ID or local path (e.g. meta-llama/Llama-2-7b-chat-hf)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / f"rsn_tune_table4_{datetime.now():%Y_%m_%d}",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--attack-config",
        type=str,
        default="gsm8k",
        help="Attack config name from benign_full_parameter_finetune/grid.yaml (default: gsm8k)",
    )
    parser.add_argument(
        "--skip-conditions",
        nargs="*",
        choices=["before", "undefended", "sn_tune", "rsn_tune", "sn_tune_orig_code", "rsn_tune_orig_code"],
        default=[],
        help="Conditions to skip (e.g. --skip-conditions before undefended)",
    )
    parser.add_argument(
        "--original-code",
        action="store_true",
        default=False,
        help="Also run SN-Tune/RSN-Tune with match_original_code=True",
    )
    parser.add_argument(
        "--original-code-suffix",
        type=str,
        default="",
        help="Suffix for orig-code config names (e.g. '_mistral' -> 'sn_tune_orig_code_mistral')",
    )
    parser.add_argument(
        "--defense-config-suffix",
        type=str,
        default="",
        help="Suffix for paper-mode defense config names (e.g. '_2k_train' -> 'sn_tune_2k_train')",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        default=False,
        help="Keep attack checkpoints (for external eval e.g. HarmBench)",
    )
    args = parser.parse_args()

    pretrained_model_path: str = args.pretrained_model_path
    results_dir: Path = args.results_dir
    random_seed: int = args.random_seed
    skip: set[str] = set(args.skip_conditions)
    keep_checkpoints: bool = args.keep_checkpoints
    run_original: bool = args.original_code
    orig_suffix: str = args.original_code_suffix
    defense_suffix: str = args.defense_config_suffix
    attack_config: str = args.attack_config

    model_alias = f"{Path(pretrained_model_path).name}_{datetime.now():%Y_%m_%d}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model config from the RSN-Tune grid config for consistency
    grid_path = DEFENSE_CONFIG_DIR / str(DefenseName.RSN_TUNE) / ConfigPath.GRID_YAML
    rsn_grid = yaml_to_dict(grid_path)
    rsn_config_dict = rsn_grid["rsn_tune"]
    assert isinstance(rsn_config_dict, dict)

    # Override attack config name if specified
    if attack_config != "gsm8k":
        attacks = rsn_config_dict.get("attacks")
        if isinstance(attacks, list):
            for attack in attacks:
                if isinstance(attack, dict):
                    attack["config_name"] = attack_config

    grid_config = DefenseGridConfig.from_dict(rsn_config_dict)
    model_config_dict = grid_config.inference_model_config
    model_config = ModelConfig.from_dict(dict(model_config_dict))
    eval_names = grid_config.defense_evals  # [strong_reject]

    all_results: dict[str, dict[str, float]] = {}

    # --- Condition 1: Before (base model, no defense, no attack) ---
    if "before" not in skip:
        print("\n" + "=" * 60)
        print("Condition 1/4: BEFORE (base model evaluation)")
        print("=" * 60)
        before_dir = results_dir / model_alias / "before"
        before_metrics = evaluate_baseline(pretrained_model_path, eval_names, model_config, before_dir)
        all_results["before"] = before_metrics
        print(f"Before metrics: {before_metrics}")
        torch.cuda.empty_cache()

    # --- Condition 2: Undefended (fine-tune on GSM8K, no defense) ---
    if "undefended" not in skip:
        print("\n" + "=" * 60)
        print("Condition 2/4: UNDEFENDED (GSM8K fine-tune, no defense)")
        print("=" * 60)
        undefended_dir = results_dir / model_alias / "undefended"
        undefended_metrics = run_attack_only(
            pretrained_model_path,
            grid_config,
            model_config_dict,
            undefended_dir,
            random_seed,
            keep_checkpoints=keep_checkpoints,
        )
        all_results["undefended"] = undefended_metrics
        print(f"Undefended metrics: {undefended_metrics}")
        torch.cuda.empty_cache()

    # --- Condition 3: SN-Tune (SN-Tune defense + GSM8K fine-tune) ---
    if "sn_tune" not in skip:
        print("\n" + "=" * 60)
        print("Condition 3/4: SN-TUNE (defense + GSM8K fine-tune)")
        print("=" * 60)
        sn_config = f"sn_tune{defense_suffix}"
        sn_metrics = run_defense_condition(
            pretrained_model_path,
            DefenseName.RSN_TUNE,
            sn_config,
            results_dir,
            random_seed,
            model_alias,
            keep_checkpoints=keep_checkpoints,
            attack_config_override=attack_config if attack_config != "gsm8k" else None,
        )
        all_results["sn_tune"] = sn_metrics
        print(f"SN-Tune metrics: {sn_metrics}")
        torch.cuda.empty_cache()

    # --- Condition 4: RSN-Tune (RSN-Tune defense + GSM8K fine-tune) ---
    if "rsn_tune" not in skip:
        print("\n" + "=" * 60)
        print("Condition 4/4: RSN-TUNE (defense + GSM8K fine-tune)")
        print("=" * 60)
        rsn_config = f"rsn_tune{defense_suffix}"
        rsn_metrics = run_defense_condition(
            pretrained_model_path,
            DefenseName.RSN_TUNE,
            rsn_config,
            results_dir,
            random_seed,
            model_alias,
            keep_checkpoints=keep_checkpoints,
            attack_config_override=attack_config if attack_config != "gsm8k" else None,
        )
        all_results["rsn_tune"] = rsn_metrics
        print(f"RSN-Tune metrics: {rsn_metrics}")
        torch.cuda.empty_cache()

    # --- Condition 5: SN-Tune (original code) ---
    sn_orig_config = f"sn_tune{defense_suffix}_orig_code{orig_suffix}"
    if run_original and "sn_tune_orig_code" not in skip:
        print("\n" + "=" * 60)
        print(f"Condition 5/6: SN-TUNE ORIG-CODE [{sn_orig_config}]")
        print("=" * 60)
        sn_orig_metrics = run_defense_condition(
            pretrained_model_path,
            DefenseName.RSN_TUNE,
            sn_orig_config,
            results_dir,
            random_seed,
            model_alias,
            keep_checkpoints=keep_checkpoints,
            attack_config_override=attack_config if attack_config != "gsm8k" else None,
        )
        all_results["sn_tune_orig_code"] = sn_orig_metrics
        print(f"SN-Tune (orig-code) metrics: {sn_orig_metrics}")
        torch.cuda.empty_cache()

    # --- Condition 6: RSN-Tune (original code) ---
    rsn_orig_config = f"rsn_tune{defense_suffix}_orig_code{orig_suffix}"
    if run_original and "rsn_tune_orig_code" not in skip:
        print("\n" + "=" * 60)
        print(f"Condition 6/6: RSN-TUNE ORIG-CODE [{rsn_orig_config}]")
        print("=" * 60)
        rsn_orig_metrics = run_defense_condition(
            pretrained_model_path,
            DefenseName.RSN_TUNE,
            rsn_orig_config,
            results_dir,
            random_seed,
            model_alias,
            keep_checkpoints=keep_checkpoints,
            attack_config_override=attack_config if attack_config != "gsm8k" else None,
        )
        all_results["rsn_tune_orig_code"] = rsn_orig_metrics
        print(f"RSN-Tune (orig-code) metrics: {rsn_orig_metrics}")
        torch.cuda.empty_cache()

    # --- Save and display results ---
    results_path = results_dir / model_alias / "table4_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    print_table(
        model_name=pretrained_model_path,
        all_results=all_results,
    )


if __name__ == "__main__":
    main()
