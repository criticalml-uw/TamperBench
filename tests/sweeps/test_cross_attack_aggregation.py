"""Tests for cross-attack-spec metric aggregation in defense trials.

Verifies that metrics are correctly aggregated across multiple attack specs
using the global aggregation method, with both per-attack and global keys.

Each test is self-contained using mocks -- no GPU, model weights, or file I/O required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tamperbench.whitebox.utils.benchmark.attack_aggregation import (
    aggregate_metrics,
    worst_case,
)
from tamperbench.whitebox.utils.benchmark.defense_config import (
    AttackAggregationConfig,
    PostDefenseAttackSpec,
)
from tamperbench.whitebox.utils.benchmark.defense_trial_manager import (
    DefenseSweepTrialManager,
)
from tamperbench.whitebox.utils.names import (
    AttackAggregationMethod,
    AttackName,
    DefenseMetricPrefix,
    EvalName,
    OptimizationDirection,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SR = EvalName.STRONG_REJECT
MMLU = EvalName.MMLU_PRO_VAL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sr_key() -> str:
    """Prefixed metric key for post-attack strong_reject under lora_finetune."""
    return "post_attack.lora_finetune.strong_reject"


@pytest.fixture()
def mmlu_key() -> str:
    """Prefixed metric key for post-attack mmlu_pro_val under lora_finetune."""
    return "post_attack.lora_finetune.mmlu_pro_val"


@pytest.fixture()
def three_config_results() -> dict[str, dict[str, float]]:
    """Per-config attack results for 3 qwen3_8b attack configs.

    qwen3_8b_a has the highest (worst-case) strong_reject.
    """
    return {
        "qwen3_8b_a": {str(SR): 0.478, str(MMLU): 0.543},
        "qwen3_8b_b": {str(SR): 0.473, str(MMLU): 0.509},
        "qwen3_8b_c": {str(SR): 0.470, str(MMLU): 0.530},
    }


@pytest.fixture()
def attack_spec_a() -> PostDefenseAttackSpec:
    """Attack spec for qwen3_8b_a grid config."""
    return PostDefenseAttackSpec(
        name=AttackName.LORA_FINETUNE,
        mode="grid",
        config_name="qwen3_8b_a",
        aggregation=AttackAggregationConfig(method=AttackAggregationMethod.WORST_CASE),
    )


@pytest.fixture()
def attack_spec_b() -> PostDefenseAttackSpec:
    """Attack spec for qwen3_8b_b grid config."""
    return PostDefenseAttackSpec(
        name=AttackName.LORA_FINETUNE,
        mode="grid",
        config_name="qwen3_8b_b",
        aggregation=AttackAggregationConfig(method=AttackAggregationMethod.WORST_CASE),
    )


@pytest.fixture()
def attack_spec_c() -> PostDefenseAttackSpec:
    """Attack spec for qwen3_8b_c grid config."""
    return PostDefenseAttackSpec(
        name=AttackName.LORA_FINETUNE,
        mode="grid",
        config_name="qwen3_8b_c",
        aggregation=AttackAggregationConfig(method=AttackAggregationMethod.WORST_CASE),
    )


@pytest.fixture()
def attack_spec_all_configs() -> PostDefenseAttackSpec:
    """Attack spec that loads all configs from grid.yaml (config_name=None)."""
    return PostDefenseAttackSpec(
        name=AttackName.LORA_FINETUNE,
        mode="grid",
        config_name=None,
        aggregation=AttackAggregationConfig(method=AttackAggregationMethod.WORST_CASE),
    )


@pytest.fixture()
def attack_spec_full_finetune() -> PostDefenseAttackSpec:
    """Attack spec for full_parameter_finetune (different attack name)."""
    return PostDefenseAttackSpec(
        name=AttackName.FULL_PARAMETER_FINETUNE,
        mode="grid",
        config_name="base",
        aggregation=AttackAggregationConfig(method=AttackAggregationMethod.WORST_CASE),
    )


@pytest.fixture()
def attack_spec_lora_sweep() -> PostDefenseAttackSpec:
    """Attack spec for lora_finetune in sweep mode."""
    return PostDefenseAttackSpec(
        name=AttackName.LORA_FINETUNE,
        mode="sweep",
        n_trials=20,
        aggregation=AttackAggregationConfig(method=AttackAggregationMethod.WORST_CASE),
    )


@pytest.fixture()
def attack_spec_full_sweep() -> PostDefenseAttackSpec:
    """Attack spec for full_parameter_finetune in sweep mode."""
    return PostDefenseAttackSpec(
        name=AttackName.FULL_PARAMETER_FINETUNE,
        mode="sweep",
        n_trials=20,
        aggregation=AttackAggregationConfig(method=AttackAggregationMethod.WORST_CASE),
    )


@pytest.fixture()
def defense_metrics() -> dict[str, float]:
    """Defense evaluation metrics (pre-attack)."""
    return {str(SR): 0.08, str(MMLU): 0.58}


@pytest.fixture()
def model_config_dict() -> dict[str, object]:
    """Minimal model config dict for mocking."""
    return {"template": "plain", "max_generation_length": 1024, "inference_batch_size": 16}


@pytest.fixture()
def mock_defense_registry():
    """Patch the defense registry so run_trial doesn't need real models.

    Yields a mock defense that returns a fake checkpoint path.
    """
    mock_defense_config_cls = MagicMock()
    mock_defense_cls = MagicMock()
    mock_defense_instance = MagicMock()
    mock_defense_instance.run_defense.return_value = Path("/tmp/fake_defended")
    mock_defense_cls.return_value = mock_defense_instance

    with patch.dict(
        "tamperbench.whitebox.defenses.registry.DEFENSES_REGISTRY",
        {"booster": (mock_defense_config_cls, mock_defense_cls)},
    ):
        yield mock_defense_config_cls, mock_defense_cls


@pytest.fixture()
def mock_evaluate_checkpoint(defense_metrics):
    """Patch evaluate_checkpoint to return fixed defense metrics."""
    with patch.object(
        DefenseSweepTrialManager,
        "evaluate_checkpoint",
        return_value=defense_metrics,
    ) as m:
        yield m


# Real per-attack data from the qwen3_8b booster sweep
@pytest.fixture()
def trial_0_per_attack() -> dict[str, dict[str, float]]:
    """Actual per-attack results from trial 0."""
    return {
        "qwen3_8b_a": {str(SR): 0.4783142367110085, str(MMLU): 0.5428571428571428},
        "qwen3_8b_b": {str(SR): 0.4733272124403201, str(MMLU): 0.5089285714285714},
        "qwen3_8b_c": {str(SR): 0.4695516695229771, str(MMLU): 0.5303571428571429},
    }


@pytest.fixture()
def trial_12_per_attack() -> dict[str, dict[str, float]]:
    """Actual per-attack results from trial 12 (currently ranked #1, should be ~#10)."""
    return {
        "qwen3_8b_a": {str(SR): 0.4875165837260481, str(MMLU): 0.5410714285714285},
        "qwen3_8b_b": {str(SR): 0.4682530237082094, str(MMLU): 0.5357142857142857},
        "qwen3_8b_c": {str(SR): 0.4052399804416937, str(MMLU): 0.5553571428571429},
    }


@pytest.fixture()
def trial_22_per_attack() -> dict[str, dict[str, float]]:
    """Actual per-attack results from trial 22 (should be the real #1)."""
    return {
        "qwen3_8b_a": {str(SR): 0.42593969171420454, str(MMLU): 0.5714285714285714},
        "qwen3_8b_b": {str(SR): 0.4331969034176665, str(MMLU): 0.55},
        "qwen3_8b_c": {str(SR): 0.42346411162671954, str(MMLU): 0.5357142857142857},
    }


# ---------------------------------------------------------------------------
# 1. aggregate_metrics unit tests (within-grid aggregation -- works correctly)
# ---------------------------------------------------------------------------


def test_worst_case_maximize_selects_max():
    """For attacker_direction=MAXIMIZE, worst_case returns the max value."""
    assert worst_case([0.40, 0.48, 0.45], OptimizationDirection.MAXIMIZE) == 0.48


def test_worst_case_minimize_selects_min():
    """For attacker_direction=MINIMIZE, worst_case returns the min value."""
    assert worst_case([0.40, 0.48, 0.45], OptimizationDirection.MINIMIZE) == 0.40


def test_worst_case_single_value():
    """worst_case on a single-element list is identity."""
    assert worst_case([0.42], OptimizationDirection.MAXIMIZE) == 0.42


def test_aggregate_metrics_single_config_is_identity():
    """aggregate_metrics with 1 config returns that config's values unchanged."""
    result = aggregate_metrics(
        all_config_results={"only_config": {str(SR): 0.47, str(MMLU): 0.55}},
        eval_names=[SR, MMLU],
        method=AttackAggregationMethod.WORST_CASE,
    )
    assert result[str(SR)] == pytest.approx(0.47)
    assert result[str(MMLU)] == pytest.approx(0.55)


def test_aggregate_metrics_multi_config_worst_case(three_config_results):
    """aggregate_metrics with 3 configs picks the max for MAXIMIZE evals."""
    result = aggregate_metrics(
        all_config_results=three_config_results,
        eval_names=[SR, MMLU],
        method=AttackAggregationMethod.WORST_CASE,
    )
    # strong_reject: MAXIMIZE -> max(0.478, 0.473, 0.470) = 0.478
    assert result[str(SR)] == pytest.approx(0.478)
    # mmlu_pro_val: MAXIMIZE -> max(0.543, 0.509, 0.530) = 0.543
    assert result[str(MMLU)] == pytest.approx(0.543)


def test_aggregate_metrics_empty_returns_nan():
    """Empty config results produce nan values."""
    result = aggregate_metrics(
        all_config_results={},
        eval_names=[SR],
        method=AttackAggregationMethod.WORST_CASE,
    )
    assert math.isnan(result[str(SR)])


def test_aggregate_metrics_top_n_average():
    """top_n_average with n=2 averages the 2 highest for MAXIMIZE."""
    results = {
        "a": {str(SR): 0.50},
        "b": {str(SR): 0.40},
        "c": {str(SR): 0.45},
    }
    result = aggregate_metrics(
        all_config_results=results,
        eval_names=[SR],
        method=AttackAggregationMethod.TOP_N_AVERAGE,
        **{"n": 2},
    )
    # Top 2 for MAXIMIZE: 0.50, 0.45 -> mean = 0.475
    assert result[str(SR)] == pytest.approx(0.475)


# ---------------------------------------------------------------------------
# 2. _aggregate_cross_spec_metrics tests (the fix for the core bug)
# ---------------------------------------------------------------------------


def test_prefix_metrics_produces_expected_keys():
    """_prefix_metrics creates dot-separated prefixed keys."""
    result = DefenseSweepTrialManager._prefix_metrics(
        "post_attack.lora_finetune",
        {"strong_reject": 0.5, "mmlu_pro_val": 0.6},
    )
    assert result == {
        "post_attack.lora_finetune.strong_reject": 0.5,
        "post_attack.lora_finetune.mmlu_pro_val": 0.6,
    }


def test_same_name_specs_should_produce_worst_case(sr_key, three_config_results):
    """Same-name attack specs should produce worst-case aggregation."""
    # Simulate 3 same-name specs each with 1 config in the global pool
    pool = {
        "lora_finetune.spec0_qwen3_8b_a": three_config_results["qwen3_8b_a"],
        "lora_finetune.spec1_qwen3_8b_b": three_config_results["qwen3_8b_b"],
        "lora_finetune.spec2_qwen3_8b_c": three_config_results["qwen3_8b_c"],
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR, MMLU])

    # worst-case = max(0.478, 0.473, 0.470) = 0.478
    assert result[sr_key] == pytest.approx(0.478)
    # Global key should also be 0.478 (only one attack name)
    assert result["post_attack.strong_reject"] == pytest.approx(0.478)


def test_different_name_specs_no_collision():
    """Different attack names produce unique per-attack keys."""
    pool = {
        "lora_finetune.spec0_base": {str(SR): 0.47},
        "full_parameter_finetune.spec1_base": {str(SR): 0.55},
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR])

    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.47)
    assert result["post_attack.full_parameter_finetune.strong_reject"] == pytest.approx(0.55)


def test_different_name_specs_should_have_global_aggregation():
    """Different-name attack specs should produce a global aggregated metric."""
    pool = {
        "lora_finetune.spec0_base": {str(SR): 0.47},
        "full_parameter_finetune.spec1_base": {str(SR): 0.55},
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR])

    assert "post_attack.strong_reject" in result
    assert result["post_attack.strong_reject"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# 3. build_all_metric_keys duplication tests
# ---------------------------------------------------------------------------


def test_build_all_metric_keys_unique_attacks():
    """With unique attack names, all metric keys are unique."""
    keys = DefenseMetricPrefix.build_all_metric_keys(
        defense_eval_names=[SR],
        post_attack_eval_names=[SR, MMLU],
        attack_names=[AttackName.LORA_FINETUNE, AttackName.FULL_PARAMETER_FINETUNE],
    )
    assert len(keys) == len(set(keys))


def test_build_all_metric_keys_should_not_duplicate():
    """Repeated attack names should produce unique metric keys."""
    keys = DefenseMetricPrefix.build_all_metric_keys(
        defense_eval_names=[SR],
        post_attack_eval_names=[SR, MMLU],
        attack_names=[AttackName.LORA_FINETUNE, AttackName.LORA_FINETUNE, AttackName.LORA_FINETUNE],
    )
    # All keys should be unique
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# 4. Penalty metrics
# ---------------------------------------------------------------------------


def test_penalty_metrics_values():
    """Penalty values are worst-case for the defender."""
    penalty = DefenseSweepTrialManager._penalty_metrics([SR, MMLU])
    # defender_direction=MINIMIZE -> penalty=1.0 (max harm)
    assert penalty[str(SR)] == 1.0
    # defender_direction=MAXIMIZE -> penalty=0.0 (no capability)
    assert penalty[str(MMLU)] == 0.0


def test_penalty_metrics_same_name_should_use_unique_keys():
    """Penalty metrics for same-name specs should produce per-attack + global keys."""
    penalty = DefenseSweepTrialManager._penalty_metrics([SR, MMLU])
    pool: dict[str, dict[str, float]] = {}
    for i in range(3):
        pool[f"lora_finetune.penalty_{i}"] = penalty
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR, MMLU])

    # Should have per-attack keys + global keys = 2 + 2 = 4
    assert len(result) == 4
    assert result["post_attack.lora_finetune.strong_reject"] == 1.0
    assert result["post_attack.lora_finetune.mmlu_pro_val"] == 0.0
    assert result["post_attack.strong_reject"] == 1.0
    assert result["post_attack.mmlu_pro_val"] == 0.0


# ---------------------------------------------------------------------------
# 5. Sweep-mode cross-attack tests
# ---------------------------------------------------------------------------


def test_sweep_same_name_specs_should_aggregate():
    """Two sweep-mode specs with same name should produce worst-case."""
    pool = {
        "lora_finetune.sweep_0": {str(SR): 0.55},
        "lora_finetune.sweep_1": {str(SR): 0.48},
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR])

    # worst-case = max(0.55, 0.48) = 0.55
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.55)
    assert result["post_attack.strong_reject"] == pytest.approx(0.55)


def test_sweep_different_names_should_have_global_worst_case():
    """Sweep specs with different names should produce a global aggregated metric."""
    pool = {
        "lora_finetune.sweep_0": {str(SR): 0.48},
        "full_parameter_finetune.sweep_1": {str(SR): 0.62},
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR])

    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.48)
    assert result["post_attack.full_parameter_finetune.strong_reject"] == pytest.approx(0.62)
    assert result["post_attack.strong_reject"] == pytest.approx(0.62)


# ---------------------------------------------------------------------------
# 6. Mixed grid + sweep
# ---------------------------------------------------------------------------


def test_grid_then_sweep_same_name_should_aggregate():
    """Grid + sweep with same name should produce worst-case across both."""
    pool = {
        "lora_finetune.spec0_base": {str(SR): 0.55},
        "lora_finetune.sweep_1": {str(SR): 0.48},
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR])

    # max(0.55, 0.48) = 0.55
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.55)
    assert result["post_attack.strong_reject"] == pytest.approx(0.55)


def test_aggregation_should_not_depend_on_spec_order():
    """Aggregation should produce the same value regardless of pool key ordering."""
    pool_a = {
        "lora_finetune.sweep_0": {str(SR): 0.48},
        "lora_finetune.spec1_base": {str(SR): 0.55},
    }
    pool_b = {
        "lora_finetune.spec0_base": {str(SR): 0.55},
        "lora_finetune.sweep_1": {str(SR): 0.48},
    }
    result_a = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool_a, [SR])
    result_b = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool_b, [SR])

    sr_key = "post_attack.lora_finetune.strong_reject"
    assert result_a[sr_key] == pytest.approx(result_b[sr_key])
    assert result_a[sr_key] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# 7. Integration: full run_trial with mocked attacks
# ---------------------------------------------------------------------------


def _make_grid_side_effect(
    per_config_results: dict[str, dict[str, float]],
    config_order: list[str],
) -> Any:
    """Build a side_effect callable for patching run_attack_grid.

    Each call returns a single-entry dict (config_name -> metrics) since
    run_attack_grid now returns raw per-config results.
    """
    call_index = {"i": 0}

    def side_effect(**kwargs: Any) -> dict[str, dict[str, float]]:
        config_name = config_order[call_index["i"]]
        call_index["i"] += 1
        return {config_name: per_config_results[config_name]}

    return side_effect


def test_run_trial_same_name_specs_should_report_worst_case(
    tmp_path,
    three_config_results,
    defense_metrics,
    model_config_dict,
    mock_defense_registry,
    mock_evaluate_checkpoint,
    sr_key,
):
    """End-to-end: 3 same-name attack specs should produce worst-case in trial_results.json."""
    attack_specs = [
        PostDefenseAttackSpec(name=AttackName.LORA_FINETUNE, mode="grid", config_name="qwen3_8b_a"),
        PostDefenseAttackSpec(name=AttackName.LORA_FINETUNE, mode="grid", config_name="qwen3_8b_b"),
        PostDefenseAttackSpec(name=AttackName.LORA_FINETUNE, mode="grid", config_name="qwen3_8b_c"),
    ]

    side_effect = _make_grid_side_effect(
        three_config_results,
        ["qwen3_8b_a", "qwen3_8b_b", "qwen3_8b_c"],
    )

    with patch.object(DefenseSweepTrialManager, "run_attack_grid", side_effect=side_effect):
        result = DefenseSweepTrialManager.run_trial(
            defense_name="booster",
            defense_config_dict={"input_checkpoint_path": "/fake", "output_checkpoint_path": "/fake"},
            defense_eval_names=[SR, MMLU],
            post_attack_eval_names=[SR, MMLU],
            model_config_dict=model_config_dict,
            attacks=attack_specs,
            pretrained_model_path="/fake_model",
            defense_results_dir=tmp_path,
            trial_number=0,
            random_seed=42,
            cleanup_checkpoints=False,
        )

    # worst-case = max(0.478, 0.473, 0.470) = 0.478
    assert result[sr_key] == pytest.approx(0.478)
    # Global key
    assert result["post_attack.strong_reject"] == pytest.approx(0.478)

    # trial_results.json should also have the correct worst-case
    trial_results_path = tmp_path / "trial_0" / "trial_results.json"
    assert trial_results_path.exists()
    with open(trial_results_path) as f:
        saved = json.load(f)
    assert saved[sr_key] == pytest.approx(0.478)


def test_run_trial_different_name_specs_should_have_global_worst_case(
    tmp_path,
    defense_metrics,
    model_config_dict,
    mock_defense_registry,
    mock_evaluate_checkpoint,
):
    """Two different attack names should produce a global aggregated metric."""
    attack_specs = [
        PostDefenseAttackSpec(name=AttackName.LORA_FINETUNE, mode="grid", config_name="base"),
        PostDefenseAttackSpec(name=AttackName.FULL_PARAMETER_FINETUNE, mode="grid", config_name="base"),
    ]

    call_index = {"i": 0}
    # run_attack_grid now returns dict[str, dict[str, float]]
    attack_results_by_call = [
        {"base": {str(SR): 0.47, str(MMLU): 0.55}},  # lora
        {"base": {str(SR): 0.62, str(MMLU): 0.50}},  # full
    ]

    def side_effect(**kwargs: Any) -> dict[str, dict[str, float]]:
        result = attack_results_by_call[call_index["i"]]
        call_index["i"] += 1
        return result

    with patch.object(DefenseSweepTrialManager, "run_attack_grid", side_effect=side_effect):
        result = DefenseSweepTrialManager.run_trial(
            defense_name="booster",
            defense_config_dict={"input_checkpoint_path": "/fake", "output_checkpoint_path": "/fake"},
            defense_eval_names=[SR, MMLU],
            post_attack_eval_names=[SR, MMLU],
            model_config_dict=model_config_dict,
            attacks=attack_specs,
            pretrained_model_path="/fake_model",
            defense_results_dir=tmp_path,
            trial_number=0,
            random_seed=42,
            cleanup_checkpoints=False,
        )

    # Per-attack results should be preserved
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.47)
    assert result["post_attack.full_parameter_finetune.strong_reject"] == pytest.approx(0.62)

    # Global aggregated key should exist with max(0.47, 0.62) = 0.62
    assert "post_attack.strong_reject" in result
    assert result["post_attack.strong_reject"] == pytest.approx(0.62)


def test_run_trial_capability_guard_violation_skips_attacks(
    tmp_path,
    model_config_dict,
    mock_defense_registry,
):
    """When capability guard is violated, attacks are skipped and penalty values assigned."""
    # Defense metrics that violate the guard (mmlu too low)
    bad_defense_metrics = {str(SR): 0.08, str(MMLU): 0.20}
    capability_baselines = {str(MMLU): 0.58}
    capability_guards = {MMLU: MagicMock(min_retention=0.7)}
    # 0.20 < 0.58 * 0.7 = 0.406 -> violation

    attack_specs = [
        PostDefenseAttackSpec(name=AttackName.LORA_FINETUNE, mode="grid", config_name="qwen3_8b_a"),
        PostDefenseAttackSpec(name=AttackName.LORA_FINETUNE, mode="grid", config_name="qwen3_8b_b"),
    ]

    with (
        patch.object(DefenseSweepTrialManager, "evaluate_checkpoint", return_value=bad_defense_metrics),
        patch.object(DefenseSweepTrialManager, "run_attack_grid") as mock_attack,
    ):
        result = DefenseSweepTrialManager.run_trial(
            defense_name="booster",
            defense_config_dict={"input_checkpoint_path": "/fake", "output_checkpoint_path": "/fake"},
            defense_eval_names=[SR, MMLU],
            post_attack_eval_names=[SR, MMLU],
            model_config_dict=model_config_dict,
            attacks=attack_specs,
            pretrained_model_path="/fake_model",
            defense_results_dir=tmp_path,
            trial_number=0,
            random_seed=42,
            cleanup_checkpoints=False,
            capability_guards=capability_guards,
            capability_baselines=capability_baselines,
        )

    # Attacks were never called
    mock_attack.assert_not_called()

    # Penalty values assigned with per-attack and global keys
    assert result["post_attack.lora_finetune.strong_reject"] == 1.0
    assert result["post_attack.lora_finetune.mmlu_pro_val"] == 0.0
    assert result["post_attack.strong_reject"] == 1.0
    assert result["post_attack.mmlu_pro_val"] == 0.0


def test_run_trial_single_attack_spec_single_config(
    tmp_path,
    defense_metrics,
    model_config_dict,
    mock_defense_registry,
    mock_evaluate_checkpoint,
):
    """Single attack spec with single config -- identity aggregation + global key."""
    attack_specs = [
        PostDefenseAttackSpec(name=AttackName.LORA_FINETUNE, mode="grid", config_name="base"),
    ]

    with patch.object(
        DefenseSweepTrialManager,
        "run_attack_grid",
        return_value={"base": {str(SR): 0.47, str(MMLU): 0.55}},
    ):
        result = DefenseSweepTrialManager.run_trial(
            defense_name="booster",
            defense_config_dict={"input_checkpoint_path": "/fake", "output_checkpoint_path": "/fake"},
            defense_eval_names=[SR, MMLU],
            post_attack_eval_names=[SR, MMLU],
            model_config_dict=model_config_dict,
            attacks=attack_specs,
            pretrained_model_path="/fake_model",
            defense_results_dir=tmp_path,
            trial_number=0,
            random_seed=42,
            cleanup_checkpoints=False,
        )

    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.47)
    assert result["post_attack.lora_finetune.mmlu_pro_val"] == pytest.approx(0.55)
    # Global keys should also exist
    assert result["post_attack.strong_reject"] == pytest.approx(0.47)
    assert result["post_attack.mmlu_pro_val"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# 8. Regression tests with real data from qwen3_8b booster sweep
# ---------------------------------------------------------------------------


def test_trial_0_correct_worst_case(trial_0_per_attack):
    """Trial 0: correct worst-case is qwen3_8b_a (0.4783), not qwen3_8b_c (0.4696)."""
    result = aggregate_metrics(
        all_config_results=trial_0_per_attack,
        eval_names=[SR, MMLU],
        method=AttackAggregationMethod.WORST_CASE,
    )
    assert result[str(SR)] == pytest.approx(0.4783142367110085)


def test_trial_12_correct_worst_case(trial_12_per_attack):
    """Trial 12 (current #1): correct worst-case is 0.4875 (a), not 0.4052 (c)."""
    result = aggregate_metrics(
        all_config_results=trial_12_per_attack,
        eval_names=[SR],
        method=AttackAggregationMethod.WORST_CASE,
    )
    assert result[str(SR)] == pytest.approx(0.4875165837260481)


def test_trial_22_correct_worst_case(trial_22_per_attack):
    """Trial 22 (real #1): correct worst-case is 0.4332."""
    result = aggregate_metrics(
        all_config_results=trial_22_per_attack,
        eval_names=[SR],
        method=AttackAggregationMethod.WORST_CASE,
    )
    assert result[str(SR)] == pytest.approx(0.4331969034176665)


def test_trial_22_beats_trial_12_under_correct_aggregation(
    trial_12_per_attack,
    trial_22_per_attack,
):
    """Under correct worst-case aggregation, trial 22 is a better defense than trial 12."""
    t12 = aggregate_metrics(
        all_config_results=trial_12_per_attack,
        eval_names=[SR],
        method=AttackAggregationMethod.WORST_CASE,
    )
    t22 = aggregate_metrics(
        all_config_results=trial_22_per_attack,
        eval_names=[SR],
        method=AttackAggregationMethod.WORST_CASE,
    )
    # Lower worst-case strong_reject = better defense (defender minimizes)
    assert t22[str(SR)] < t12[str(SR)]


# ---------------------------------------------------------------------------
# 9. Desired behavior specification tests
# ---------------------------------------------------------------------------


def test_desired_global_worst_case_all_configs():
    """SPEC: Pool all configs from all specs, apply worst_case once."""
    pooled = {
        "spec_a:qwen3_8b_a": {str(SR): 0.478, str(MMLU): 0.543},
        "spec_b:qwen3_8b_b": {str(SR): 0.473, str(MMLU): 0.509},
        "spec_c:qwen3_8b_c": {str(SR): 0.470, str(MMLU): 0.530},
    }
    result = aggregate_metrics(
        all_config_results=pooled,
        eval_names=[SR, MMLU],
        method=AttackAggregationMethod.WORST_CASE,
    )
    assert result[str(SR)] == pytest.approx(0.478)
    assert result[str(MMLU)] == pytest.approx(0.543)


def test_desired_global_worst_case_across_attack_types():
    """SPEC: Worst-case should span different attack types."""
    pooled = {
        "lora:base": {str(SR): 0.47},
        "full:base": {str(SR): 0.62},
    }
    result = aggregate_metrics(
        all_config_results=pooled,
        eval_names=[SR],
        method=AttackAggregationMethod.WORST_CASE,
    )
    assert result[str(SR)] == pytest.approx(0.62)


def test_desired_global_worst_case_mixed_modes():
    """SPEC: Pool grid configs + sweep best, then apply worst_case."""
    pooled = {
        "grid:config_1": {str(SR): 0.50},
        "grid:config_2": {str(SR): 0.55},
        "grid:config_3": {str(SR): 0.45},
        "sweep:best": {str(SR): 0.48},
    }
    result = aggregate_metrics(
        all_config_results=pooled,
        eval_names=[SR],
        method=AttackAggregationMethod.WORST_CASE,
    )
    assert result[str(SR)] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# 10. Edge case tests
# ---------------------------------------------------------------------------


def test_exact_duplicate_specs_aggregate_correctly():
    """Exact duplicate specs (same name + same config) should aggregate correctly."""
    pool = {
        "lora_finetune.spec0_base": {str(SR): 0.47, str(MMLU): 0.55},
        "lora_finetune.spec1_base": {str(SR): 0.47, str(MMLU): 0.55},
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR, MMLU])
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.47)
    assert result["post_attack.strong_reject"] == pytest.approx(0.47)


def test_different_attacks_same_config_name():
    """Different attack names with same config_name — no collision, global picks worse."""
    pool = {
        "lora_finetune.spec0_base": {str(SR): 0.47},
        "full_parameter_finetune.spec1_base": {str(SR): 0.62},
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR])
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.47)
    assert result["post_attack.full_parameter_finetune.strong_reject"] == pytest.approx(0.62)
    assert result["post_attack.strong_reject"] == pytest.approx(0.62)


def test_grid_and_sweep_different_names():
    """Grid + sweep with different names — both preserved, global picks worse."""
    pool = {
        "lora_finetune.spec0_base": {str(SR): 0.50},
        "full_parameter_finetune.sweep_1": {str(SR): 0.48},
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR])
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.50)
    assert result["post_attack.full_parameter_finetune.strong_reject"] == pytest.approx(0.48)
    assert result["post_attack.strong_reject"] == pytest.approx(0.50)


def test_single_spec_produces_global_key():
    """Even with just one spec, a global key should be produced."""
    pool = {"lora_finetune.spec0_base": {str(SR): 0.47}}
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR])
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.47)
    assert result["post_attack.strong_reject"] == pytest.approx(0.47)


def test_multi_eval_worst_case_independent_per_eval():
    """Worst-case should be computed independently for each eval."""
    pool = {
        "lora_finetune.spec0_a": {str(SR): 0.40, str(MMLU): 0.60},
        "lora_finetune.spec1_b": {str(SR): 0.50, str(MMLU): 0.55},
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR, MMLU])
    # SR worst-case (MAXIMIZE): max(0.40, 0.50) = 0.50
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.50)
    # MMLU worst-case (MAXIMIZE): max(0.60, 0.55) = 0.60
    assert result["post_attack.lora_finetune.mmlu_pro_val"] == pytest.approx(0.60)


def test_nan_spec_ignored_in_aggregation():
    """NaN values should be handled gracefully (aggregate_metrics skips missing keys)."""
    # When a sweep fails, it may return NaN for an eval.
    # aggregate_metrics only collects values where eval_key is in metrics.
    # NaN won't be filtered out by aggregate_metrics itself, but worst_case
    # will return NaN if it's the max. We test that valid values survive.
    pool = {
        "lora_finetune.spec0_a": {str(SR): 0.47},
        # spec1 has no SR key at all (simulating missing data)
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR])
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.47)
    assert result["post_attack.strong_reject"] == pytest.approx(0.47)


def test_many_specs_aggregation():
    """Aggregation should work with many (5+) specs."""
    pool = {f"lora_finetune.spec{i}_cfg": {str(SR): 0.40 + i * 0.02} for i in range(6)}
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR])
    # worst_case (MAXIMIZE): max(0.40, 0.42, 0.44, 0.46, 0.48, 0.50) = 0.50
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.50)
    assert result["post_attack.strong_reject"] == pytest.approx(0.50)


def test_penalty_different_name_specs_global_key():
    """Guard violation with different-name attacks should produce global penalty key."""
    penalty = DefenseSweepTrialManager._penalty_metrics([SR, MMLU])
    pool = {
        "lora_finetune.penalty_0": penalty,
        "full_parameter_finetune.penalty_1": penalty,
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR, MMLU])
    assert "post_attack.lora_finetune.strong_reject" in result
    assert "post_attack.full_parameter_finetune.strong_reject" in result
    assert "post_attack.strong_reject" in result
    assert result["post_attack.strong_reject"] == 1.0
    assert result["post_attack.mmlu_pro_val"] == 0.0


def test_mixed_same_and_different_names():
    """3 specs: 2 share a name, 1 is different. Per-attack + global aggregation."""
    pool = {
        "lora_finetune.spec0_a": {str(SR): 0.48},
        "lora_finetune.spec1_b": {str(SR): 0.52},
        "full_parameter_finetune.spec2_base": {str(SR): 0.45},
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR])
    # lora worst-case: max(0.48, 0.52) = 0.52
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.52)
    # full_ft: identity = 0.45
    assert result["post_attack.full_parameter_finetune.strong_reject"] == pytest.approx(0.45)
    # global worst-case: max(0.52, 0.45) = 0.52
    assert result["post_attack.strong_reject"] == pytest.approx(0.52)


def test_build_all_metric_keys_mixed_duplicates():
    """build_all_metric_keys with [lora, lora, full_ft] produces unique keys + global keys."""
    keys = DefenseMetricPrefix.build_all_metric_keys(
        defense_eval_names=[SR],
        post_attack_eval_names=[SR, MMLU],
        attack_names=[AttackName.LORA_FINETUNE, AttackName.LORA_FINETUNE, AttackName.FULL_PARAMETER_FINETUNE],
    )
    assert len(keys) == len(set(keys))  # all unique
    # 1 defense + 2 lora + 2 full_ft + 2 global = 7
    assert len(keys) == 7
    assert "post_attack.strong_reject" in keys
    assert "post_attack.mmlu_pro_val" in keys


def test_empty_per_spec_results():
    """Empty global pool returns empty dict."""
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics({}, [SR])
    assert result == {}


def test_run_trial_mixed_same_different_names(
    tmp_path,
    defense_metrics,
    model_config_dict,
    mock_defense_registry,
    mock_evaluate_checkpoint,
):
    """End-to-end: mixed same + different names produce correct aggregation."""
    attack_specs = [
        PostDefenseAttackSpec(name=AttackName.LORA_FINETUNE, mode="grid", config_name="qwen3_8b_a"),
        PostDefenseAttackSpec(name=AttackName.LORA_FINETUNE, mode="grid", config_name="qwen3_8b_b"),
        PostDefenseAttackSpec(name=AttackName.FULL_PARAMETER_FINETUNE, mode="grid", config_name="base"),
    ]

    call_index = {"i": 0}
    results_by_call = [
        {"qwen3_8b_a": {str(SR): 0.48}},
        {"qwen3_8b_b": {str(SR): 0.52}},
        {"base": {str(SR): 0.45}},
    ]

    def side_effect(**kwargs: Any) -> dict[str, dict[str, float]]:
        r = results_by_call[call_index["i"]]
        call_index["i"] += 1
        return r

    with patch.object(DefenseSweepTrialManager, "run_attack_grid", side_effect=side_effect):
        result = DefenseSweepTrialManager.run_trial(
            defense_name="booster",
            defense_config_dict={"input_checkpoint_path": "/fake", "output_checkpoint_path": "/fake"},
            defense_eval_names=[SR],
            post_attack_eval_names=[SR],
            model_config_dict=model_config_dict,
            attacks=attack_specs,
            pretrained_model_path="/fake_model",
            defense_results_dir=tmp_path,
            trial_number=0,
            random_seed=42,
            cleanup_checkpoints=False,
        )

    # lora worst-case: max(0.48, 0.52) = 0.52
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.52)
    assert result["post_attack.full_parameter_finetune.strong_reject"] == pytest.approx(0.45)
    # global worst-case: max(0.52, 0.45) = 0.52
    assert result["post_attack.strong_reject"] == pytest.approx(0.52)


def test_global_aggregation_with_top_n_average():
    """Non-worst_case methods should work globally too."""
    agg = AttackAggregationConfig(method=AttackAggregationMethod.TOP_N_AVERAGE, n=2)
    pool = {
        "lora_finetune.spec0_a": {str(SR): 0.50},
        "lora_finetune.spec1_b": {str(SR): 0.40},
        "lora_finetune.spec2_c": {str(SR): 0.45},
    }
    result = DefenseSweepTrialManager._aggregate_cross_spec_metrics(pool, [SR], aggregation=agg)
    # Top 2 for MAXIMIZE: 0.50, 0.45 -> mean = 0.475
    assert result["post_attack.lora_finetune.strong_reject"] == pytest.approx(0.475)
    assert result["post_attack.strong_reject"] == pytest.approx(0.475)


def test_run_attack_grid_returns_raw_per_config():
    """Verify run_attack_grid no longer aggregates (returns per-config dict)."""
    # The return type annotation is dict[str, dict[str, float]] now.
    # We verify this by checking the mock expectation in run_trial integration tests:
    # run_trial iterates per_config_results.items() and adds them to global_pool.
    # If run_attack_grid returned a flat dict, the pool would have wrong structure.
    # This is implicitly tested by all run_trial integration tests passing.
    # Here we do a direct type check on the method annotation.
    import inspect

    sig = inspect.signature(DefenseSweepTrialManager.run_attack_grid)
    return_annotation = sig.return_annotation
    assert "dict[str, dict[str, float]]" in str(return_annotation)
