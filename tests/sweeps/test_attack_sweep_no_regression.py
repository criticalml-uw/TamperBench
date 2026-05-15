"""Regression tests for the attack sweep (optuna_single.py) flow.

The attack sweep (scripts/whitebox/optuna_single.py) is NOT affected by
the cross-attack aggregation bug because:
  1. Each attack runs its own independent Optuna study
  2. Each trial executes ONE attack with ONE config (no multi-config grid)
  3. No dict.update() collision -- metrics use bare eval names (not prefixed)
  4. No cross-attack merging of results

These tests verify that the attack sweep flow remains correct and
guard against future regressions that might introduce similar bugs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tamperbench.whitebox.utils.benchmark.runners import run_optuna_sweep
from tamperbench.whitebox.utils.benchmark.trial_manager import SweepTrialManager
from tamperbench.whitebox.utils.names import (
    AttackName,
    EvalName,
    OptimizationDirection,
    OptunaUserAttrs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = EvalName.STRONG_REJECT
MMLU = EvalName.MMLU_PRO_VAL


@pytest.fixture()
def study_paths(tmp_path) -> Any:
    """Mock StudyPaths object with real temp directory."""
    paths = MagicMock()
    sweep_dir = tmp_path / "sweep_results"
    sweep_dir.mkdir()
    paths.sweep_results_dir = sweep_dir
    paths.storage_path = sweep_dir / "study.db"
    paths.storage_url = f"sqlite:///{sweep_dir / 'study.db'}"
    paths.study_name = "test_study"
    paths.ensure_dirs = MagicMock()
    return paths


@pytest.fixture()
def base_attack_config() -> dict[str, object]:
    """Minimal base config for a lora_finetune attack."""
    return {
        "per_device_train_batch_size": 8,
        "learning_rate": 0.0001,
        "num_train_epochs": 1,
    }


@pytest.fixture()
def sweep_space() -> dict[str, object]:
    """Minimal sweep search space."""
    return {
        "learning_rate": {
            "type": "float",
            "low": 1e-5,
            "high": 1e-3,
            "log": True,
        },
    }


# ---------------------------------------------------------------------------
# 1. SweepTrialManager.run_trial returns bare eval names (no prefixes)
# ---------------------------------------------------------------------------


def test_run_trial_returns_bare_eval_names():
    """Attack trial returns {eval_name: value} with NO dot-prefixed keys."""
    mock_attack_config_cls = MagicMock()
    mock_attack_cls = MagicMock()

    mock_attacker = MagicMock()
    mock_results_df = MagicMock()
    mock_attacker.benchmark.return_value = mock_results_df
    mock_attack_cls.return_value = mock_attacker

    with patch.dict(
        "tamperbench.whitebox.attacks.registry.ATTACKS_REGISTRY",
        {"lora_finetune": (mock_attack_config_cls, mock_attack_cls)},
    ):
        # Mock the eval registry to return a fixed value
        mock_eval_cls = MagicMock()
        mock_eval_cls.load_result_objective.return_value = 0.75

        with patch.dict(
            "tamperbench.whitebox.evals.registry.EVALS_REGISTRY",
            {SR: mock_eval_cls, MMLU: mock_eval_cls},
        ):
            result = SweepTrialManager.run_trial(
                attack_name=AttackName.LORA_FINETUNE,
                attack_config_dict={"learning_rate": 0.001},
                eval_names=[SR, MMLU],
                pretrained_model_path="/fake/model",
                attack_results_dir=Path("/tmp/test_attack"),
                trial_number=0,
                random_seed=42,
            )

    # Keys are bare eval names -- no "post_attack." prefix
    assert str(SR) in result
    assert str(MMLU) in result
    assert not any("post_attack" in key for key in result)
    assert not any("defense" in key for key in result)


def test_run_trial_each_eval_gets_own_value():
    """Each eval is loaded independently from the benchmark results."""
    mock_attack_config_cls = MagicMock()
    mock_attack_cls = MagicMock()
    mock_attacker = MagicMock()
    mock_results_df = MagicMock()
    mock_attacker.benchmark.return_value = mock_results_df
    mock_attack_cls.return_value = mock_attacker

    mock_sr_eval = MagicMock()
    mock_sr_eval.load_result_objective.return_value = 0.65
    mock_mmlu_eval = MagicMock()
    mock_mmlu_eval.load_result_objective.return_value = 0.55

    with (
        patch.dict(
            "tamperbench.whitebox.attacks.registry.ATTACKS_REGISTRY",
            {"lora_finetune": (mock_attack_config_cls, mock_attack_cls)},
        ),
        patch.dict(
            "tamperbench.whitebox.evals.registry.EVALS_REGISTRY",
            {SR: mock_sr_eval, MMLU: mock_mmlu_eval},
        ),
    ):
        result = SweepTrialManager.run_trial(
            attack_name=AttackName.LORA_FINETUNE,
            attack_config_dict={"learning_rate": 0.001},
            eval_names=[SR, MMLU],
            pretrained_model_path="/fake/model",
            attack_results_dir=Path("/tmp/test_attack"),
            trial_number=0,
        )

    assert result[str(SR)] == pytest.approx(0.65)
    assert result[str(MMLU)] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# 2. run_optuna_sweep stores metrics correctly per trial
# ---------------------------------------------------------------------------


def test_optuna_sweep_stores_metrics_in_user_attrs(study_paths, base_attack_config, sweep_space):
    """Each Optuna trial stores its eval_metrics in user_attrs without collisions."""
    call_count = {"n": 0}
    trial_results = [
        {str(SR): 0.65, str(MMLU): 0.55},
        {str(SR): 0.70, str(MMLU): 0.52},
        {str(SR): 0.60, str(MMLU): 0.58},
    ]

    def fake_objective(trial: Any, merged_config: dict[str, object]) -> dict[str, float]:
        result = trial_results[call_count["n"]]
        call_count["n"] += 1
        return result

    study = run_optuna_sweep(
        study_paths=study_paths,
        sweep_space=sweep_space,
        base_config=base_attack_config,
        direction="maximize",
        objective_fn=fake_objective,
        primary_metric_key=str(SR),
        n_trials=3,
        random_seed=42,
        top_n=3,
        eval_names=[SR, MMLU],
        base_config_name="base",
    )

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    assert len(completed) == 3

    # Each trial has its own metrics -- no cross-trial contamination
    for trial in completed:
        metrics = trial.user_attrs.get(OptunaUserAttrs.EVAL_METRICS)
        assert isinstance(metrics, dict)
        assert str(SR) in metrics
        assert str(MMLU) in metrics


def test_optuna_sweep_primary_metric_returned_to_optuna(study_paths, base_attack_config, sweep_space):
    """Optuna sees the primary metric value from each trial."""
    results = [
        {str(SR): 0.65, str(MMLU): 0.55},
        {str(SR): 0.70, str(MMLU): 0.52},
    ]
    call_idx = {"i": 0}

    def fake_objective(trial: Any, merged_config: dict[str, object]) -> dict[str, float]:
        r = results[call_idx["i"]]
        call_idx["i"] += 1
        return r

    study = run_optuna_sweep(
        study_paths=study_paths,
        sweep_space=sweep_space,
        base_config=base_attack_config,
        direction="maximize",
        objective_fn=fake_objective,
        primary_metric_key=str(SR),
        n_trials=2,
        random_seed=42,
        top_n=2,
        eval_names=[SR, MMLU],
        base_config_name="base",
    )

    completed = sorted(
        [t for t in study.trials if t.state.name == "COMPLETE"],
        key=lambda t: t.number,
    )
    assert completed[0].values[0] == pytest.approx(0.65)
    assert completed[1].values[0] == pytest.approx(0.70)


def test_optuna_sweep_best_trial_is_correct(study_paths, base_attack_config, sweep_space):
    """Best trial is selected correctly based on direction."""
    results = [
        {str(SR): 0.60},
        {str(SR): 0.80},
        {str(SR): 0.70},
    ]
    call_idx = {"i": 0}

    def fake_objective(trial: Any, merged_config: dict[str, object]) -> dict[str, float]:
        r = results[call_idx["i"]]
        call_idx["i"] += 1
        return r

    study = run_optuna_sweep(
        study_paths=study_paths,
        sweep_space=sweep_space,
        base_config=base_attack_config,
        direction="maximize",
        objective_fn=fake_objective,
        primary_metric_key=str(SR),
        n_trials=3,
        random_seed=42,
        top_n=1,
        eval_names=[SR],
        base_config_name="base",
    )

    assert study.best_value == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# 3. Independent attacks don't interfere with each other
# ---------------------------------------------------------------------------


def test_sequential_attack_sweeps_independent(tmp_path, base_attack_config, sweep_space):
    """Running two attack sweeps sequentially produces independent studies."""
    study_a = MagicMock()
    study_a.sweep_results_dir = tmp_path / "lora"
    study_a.sweep_results_dir.mkdir()
    study_a.storage_path = study_a.sweep_results_dir / "study.db"
    study_a.storage_url = f"sqlite:///{study_a.storage_path}"
    study_a.study_name = "lora_study"
    study_a.ensure_dirs = MagicMock()

    study_b = MagicMock()
    study_b.sweep_results_dir = tmp_path / "full"
    study_b.sweep_results_dir.mkdir()
    study_b.storage_path = study_b.sweep_results_dir / "study.db"
    study_b.storage_url = f"sqlite:///{study_b.storage_path}"
    study_b.study_name = "full_study"
    study_b.ensure_dirs = MagicMock()

    lora_results = [{str(SR): 0.65}, {str(SR): 0.70}]
    full_results = [{str(SR): 0.80}, {str(SR): 0.85}]

    lora_idx = {"i": 0}
    full_idx = {"i": 0}

    def lora_objective(trial: Any, config: dict[str, object]) -> dict[str, float]:
        r = lora_results[lora_idx["i"]]
        lora_idx["i"] += 1
        return r

    def full_objective(trial: Any, config: dict[str, object]) -> dict[str, float]:
        r = full_results[full_idx["i"]]
        full_idx["i"] += 1
        return r

    result_a = run_optuna_sweep(
        study_paths=study_a,
        sweep_space=sweep_space,
        base_config=base_attack_config,
        direction="maximize",
        objective_fn=lora_objective,
        primary_metric_key=str(SR),
        n_trials=2,
        random_seed=42,
        top_n=1,
        eval_names=[SR],
        base_config_name="base",
    )

    result_b = run_optuna_sweep(
        study_paths=study_b,
        sweep_space=sweep_space,
        base_config=base_attack_config,
        direction="maximize",
        objective_fn=full_objective,
        primary_metric_key=str(SR),
        n_trials=2,
        random_seed=42,
        top_n=1,
        eval_names=[SR],
        base_config_name="base",
    )

    # Each study has its own best, independent of the other
    assert result_a.best_value == pytest.approx(0.70)
    assert result_b.best_value == pytest.approx(0.85)

    # No cross-contamination
    a_metrics = [
        t.user_attrs[OptunaUserAttrs.EVAL_METRICS][str(SR)] for t in result_a.trials if t.state.name == "COMPLETE"
    ]
    b_metrics = [
        t.user_attrs[OptunaUserAttrs.EVAL_METRICS][str(SR)] for t in result_b.trials if t.state.name == "COMPLETE"
    ]
    assert set(a_metrics) == {0.65, 0.70}
    assert set(b_metrics) == {0.80, 0.85}


# ---------------------------------------------------------------------------
# 4. best.json output is correct for attack sweeps
# ---------------------------------------------------------------------------


def test_best_json_written_correctly(study_paths, base_attack_config, sweep_space):
    """best.json contains correct structure and values for attack sweep."""
    results = [
        {str(SR): 0.60, str(MMLU): 0.55},
        {str(SR): 0.80, str(MMLU): 0.50},
        {str(SR): 0.70, str(MMLU): 0.52},
    ]
    idx = {"i": 0}

    def fake_obj(trial: Any, config: dict[str, object]) -> dict[str, float]:
        r = results[idx["i"]]
        idx["i"] += 1
        return r

    run_optuna_sweep(
        study_paths=study_paths,
        sweep_space=sweep_space,
        base_config=base_attack_config,
        direction="maximize",
        objective_fn=fake_obj,
        primary_metric_key=str(SR),
        n_trials=3,
        random_seed=42,
        top_n=2,
        eval_names=[SR, MMLU],
        base_config_name="base",
    )

    best_json_path = study_paths.sweep_results_dir / "best.json"
    assert best_json_path.exists()

    with open(best_json_path) as f:
        best = json.load(f)

    assert "top_trials" in best
    assert len(best["top_trials"]) == 2

    # Top trial should be the one with SR=0.80 (maximize)
    top_1 = best["top_trials"][0]
    assert top_1["rank"] == 1
    # Values list should have SR and MMLU in order
    assert len(top_1["values"]) == 2
    assert top_1["eval_names"] == [str(SR), str(MMLU)]

    # No duplicate eval_names (unlike the defense sweep bug)
    assert len(top_1["eval_names"]) == len(set(top_1["eval_names"]))


# ---------------------------------------------------------------------------
# 5. Attack sweep uses bare metric keys (not prefixed) -- guard against regression
# ---------------------------------------------------------------------------


def test_attack_objective_returns_bare_keys():
    """The attack objective function returns bare eval names, not prefixed keys.

    This is critical -- if someone accidentally adds prefixing to the attack
    path, the primary_metric_key lookup would break.
    """
    mock_attack_config_cls = MagicMock()
    mock_attack_cls = MagicMock()
    mock_attacker = MagicMock()
    mock_results_df = MagicMock()
    mock_attacker.benchmark.return_value = mock_results_df
    mock_attack_cls.return_value = mock_attacker

    mock_eval = MagicMock()
    mock_eval.load_result_objective.return_value = 0.75

    with (
        patch.dict(
            "tamperbench.whitebox.attacks.registry.ATTACKS_REGISTRY",
            {"lora_finetune": (mock_attack_config_cls, mock_attack_cls)},
        ),
        patch.dict(
            "tamperbench.whitebox.evals.registry.EVALS_REGISTRY",
            {SR: mock_eval},
        ),
    ):
        result = SweepTrialManager.run_trial(
            attack_name=AttackName.LORA_FINETUNE,
            attack_config_dict={"learning_rate": 0.001},
            eval_names=[SR],
            pretrained_model_path="/fake",
            attack_results_dir=Path("/tmp/test"),
            trial_number=0,
        )

    # primary_metric_key in optuna_single.py is str(sweep_config.evals[0])
    # which is "strong_reject" (bare). This must exist in the result dict.
    primary_key = str(SR)  # = "strong_reject"
    assert primary_key in result
    assert result[primary_key] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 6. sorted_completed_trials direction correctness
# ---------------------------------------------------------------------------


def test_sorted_completed_trials_maximize():
    """For maximize direction, best trial (highest value) comes first."""
    import optuna

    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(values=[0.5], state=optuna.trial.TrialState.COMPLETE))
    study.add_trial(optuna.trial.create_trial(values=[0.8], state=optuna.trial.TrialState.COMPLETE))
    study.add_trial(optuna.trial.create_trial(values=[0.3], state=optuna.trial.TrialState.COMPLETE))

    sorted_trials = SweepTrialManager.sorted_completed_trials(study, "maximize")
    values = [t.values[0] for t in sorted_trials]
    assert values == [0.8, 0.5, 0.3]


def test_sorted_completed_trials_minimize():
    """For minimize direction, best trial (lowest value) comes first."""
    import optuna

    study = optuna.create_study(direction="minimize")
    study.add_trial(optuna.trial.create_trial(values=[0.5], state=optuna.trial.TrialState.COMPLETE))
    study.add_trial(optuna.trial.create_trial(values=[0.8], state=optuna.trial.TrialState.COMPLETE))
    study.add_trial(optuna.trial.create_trial(values=[0.3], state=optuna.trial.TrialState.COMPLETE))

    sorted_trials = SweepTrialManager.sorted_completed_trials(study, "minimize")
    values = [t.values[0] for t in sorted_trials]
    assert values == [0.3, 0.5, 0.8]


def test_sorted_completed_trials_skips_pruned():
    """Pruned/failed trials are excluded from sorting."""
    import optuna

    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(values=[0.5], state=optuna.trial.TrialState.COMPLETE))
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED))
    study.add_trial(optuna.trial.create_trial(values=[0.8], state=optuna.trial.TrialState.COMPLETE))

    sorted_trials = SweepTrialManager.sorted_completed_trials(study, "maximize")
    assert len(sorted_trials) == 2
    assert sorted_trials[0].values[0] == 0.8


# ---------------------------------------------------------------------------
# 7. Error handling in attack sweep
# ---------------------------------------------------------------------------


def test_optuna_sweep_handles_trial_failure(study_paths, base_attack_config, sweep_space):
    """When a trial raises an exception, it's pruned and the sweep continues."""
    call_idx = {"i": 0}

    def flaky_objective(trial: Any, config: dict[str, object]) -> dict[str, float]:
        i = call_idx["i"]
        call_idx["i"] += 1
        if i == 1:
            raise RuntimeError("GPU OOM")
        return {str(SR): 0.60 + i * 0.05}

    study = run_optuna_sweep(
        study_paths=study_paths,
        sweep_space=sweep_space,
        base_config=base_attack_config,
        direction="maximize",
        objective_fn=flaky_objective,
        primary_metric_key=str(SR),
        n_trials=3,
        random_seed=42,
        top_n=1,
        eval_names=[SR],
        base_config_name="base",
    )

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]

    assert len(completed) >= 2
    assert len(pruned) >= 1

    # Pruned trial has failure info
    failed = pruned[0]
    assert "failure" in failed.user_attrs
    assert "GPU OOM" in failed.user_attrs["failure"]["error"]


# ---------------------------------------------------------------------------
# 8. No metric key collision possible in attack sweep
# ---------------------------------------------------------------------------


def test_attack_sweep_metric_keys_are_bare_and_unique(study_paths, base_attack_config, sweep_space):
    """Attack sweep metrics use bare eval names -- collision impossible."""
    eval_names = [SR, MMLU]

    results = [{str(SR): 0.65, str(MMLU): 0.55}]
    idx = {"i": 0}

    def obj(trial: Any, config: dict[str, object]) -> dict[str, float]:
        r = results[idx["i"]]
        idx["i"] += 1
        return r

    study = run_optuna_sweep(
        study_paths=study_paths,
        sweep_space=sweep_space,
        base_config=base_attack_config,
        direction="maximize",
        objective_fn=obj,
        primary_metric_key=str(SR),
        n_trials=1,
        random_seed=42,
        top_n=1,
        eval_names=eval_names,
        base_config_name="base",
    )

    trial = study.trials[0]
    metrics = trial.user_attrs[OptunaUserAttrs.EVAL_METRICS]

    # Exactly one key per eval, no duplicates, no prefixes
    assert set(metrics.keys()) == {str(SR), str(MMLU)}
    assert len(metrics) == 2
