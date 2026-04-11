# Defenses Guide

This guide covers how to run and benchmark alignment defenses. For attack benchmarking, see [USAGE.md](USAGE.md).

## Use Cases

| Use Case                                 | Approach              | When to Use                                  |
| ---------------------------------------- | --------------------- | -------------------------------------------- |
| Apply a defense to a model               | Direct Python API     | Testing, debugging, custom workflows         |
| Benchmark a defense with fixed configs   | `defense_grid.py`     | Pre-defined hyperparameters, quick tests     |
| Sweep defense hyperparameters            | `defense_sweep.py`    | Research, finding optimal defense settings   |

## Direct Python API

### Applying a Defense

```python
from pathlib import Path
from tamperbench.whitebox.defenses.crl import CRL, CRLConfig

config = CRLConfig(
    input_checkpoint_path=Path("meta-llama/Llama-3.1-8B-Instruct"),
    output_checkpoint_path=Path("results/defended_model"),
    num_samples=10000,
    num_steps=1100,
    batch_size=16,
    learning_rate=1e-5,
    representation_layers=tuple(range(20, 32)),
    lora_r=16,
    lora_alpha=16,
)

defense = CRL(defense_config=config)
defended_path = defense.run_defense()
print(f"Defended model saved to: {defended_path}")
```

Available defenses are listed in the `DefenseName` enum in `src/tamperbench/whitebox/utils/names.py`. All defenses follow the same interface: configure with a dataclass inheriting `AlignmentDefenseConfig`, then call `run_defense()`.

## Defense Benchmark Scripts

The defense benchmark pipeline runs: **defend -> eval defense checkpoint -> attack defended model -> eval post-attack checkpoint -> cleanup**. This measures both how well the defense preserves safety and whether it survives adversarial fine-tuning.

### Config Structure

Each defense has two config files under `configs/whitebox/defenses/<defense>/`:

1. **`grid.yaml`** -- named config entries with full hyperparameters + orchestration metadata
2. **`single_objective_sweep.yaml`** -- hyperparameter search space + shared orchestration metadata (no defense hparams, just the sweep ranges)

Both files share the same orchestration metadata fields, which are stripped before passing to the defense's `AlignmentDefenseConfig`.

#### `grid.yaml`

Each top-level key is a named config (conventionally `base`). The config contains orchestration fields and defense-specific hyperparameters mixed together:

```yaml
base: &base_cfg
    # --- Orchestration fields (stripped before passing to defense) ---
    model_config:
        template: plain                # model template for inference
        max_generation_length: 1024
        inference_batch_size: 16
    defense_evals: [strong_reject, mmlu_pro_val]     # evals on defended checkpoint
    post_attack_evals: [strong_reject, mmlu_pro_val]  # evals on post-attack checkpoint
    attacks:
        - name: lora_finetune          # attack from AttackName enum
          mode: grid                   # "grid" or "sweep"
          config_name: base            # which config from attack's grid.yaml (null = all)

    # --- Defense-specific hyperparameters (passed to AlignmentDefenseConfig) ---
    alpha: 0.5
    learning_rate: 1.0e-5
    lora_r: 16
    # ... (varies by defense)
```

The `attacks` list references attack configs from `configs/whitebox/attacks/<attack>/grid.yaml`. For example, `name: lora_finetune` with `config_name: base` loads the `base` entry from `configs/whitebox/attacks/lora_finetune/grid.yaml`. Setting `config_name: null` runs all configs in that file -- useful for stress-testing against multiple attack configurations, and required for aggregation methods that operate over multiple configs.

Model-specific overrides can be placed in separate directories (e.g., `configs/whitebox/defenses_llama/crl/grid.yaml`), following the same pattern as `attacks_llama/`, `attacks_qwen/`.

#### `single_objective_sweep.yaml`

Contains only orchestration metadata and the hyperparameter search space -- no defense hparams (those come from `grid.yaml`'s base config, which is loaded and used as the sweep's starting point):

```yaml
defense_evals: [strong_reject, mmlu_pro_val]
post_attack_evals: [strong_reject, mmlu_pro_val]
model_config:
    template: plain
    max_generation_length: 1024
    inference_batch_size: 16
attacks:
    - name: lora_finetune
      mode: grid
      config_name: base

# Optional: skip attack phase if defense hurts capability too much
capability_guards:
    mmlu_pro_val:
        min_retention: 0.8

# Optuna search space for defense hyperparameters
sweep:
    alpha:
        type: float
        low: 0.1
        high: 1.0
    learning_rate:
        type: float
        low: 1.0e-6
        high: 1.0e-3
        log: true
    lora_r:
        choices: [8, 16, 32]
```

The sweep ranges override keys from the base config loaded from `grid.yaml`. The search space syntax matches attack sweeps (see `configs/whitebox/attacks/<attack>/single_objective_sweep.yaml` for examples).

### Orchestration Fields Reference

These fields are present in both `grid.yaml` entries and `single_objective_sweep.yaml`:

- **`model_config`** -- inference settings passed to `ModelConfig`. Used for evaluation, not training.
- **`defense_evals`** -- evaluations run on the defended checkpoint (pre-attack). Measures safety + utility preservation. Values must be from the `EvalName` enum.
- **`post_attack_evals`** -- evaluations run on post-attack checkpoints. Measures whether the defense survived. Values must be from the `EvalName` enum.
- **`attacks`** -- list of `PostDefenseAttackSpec` entries defining which attacks to stress-test with:
  - `name` -- attack from `AttackName` enum (e.g., `lora_finetune`, `full_parameter_finetune`)
  - `mode` -- `"grid"` (fixed configs from attack's `grid.yaml`) or `"sweep"` (inner Optuna sweep from attacker's perspective)
  - `config_name` -- for grid mode: which config key to use from the attack's `grid.yaml`. Set to `null` to run all configs. Defaults to `"base"`.
  - `configs_dir` -- optional override for attack config directory (defaults to `configs/whitebox/attacks`)
  - `n_trials` -- for sweep mode: number of inner Optuna trials (default: 10). The inner sweep loads the attack's `single_objective_sweep.yaml` search space and `grid.yaml` base config.
  - `aggregation` -- how to aggregate metrics across grid configs (see [Post-Attack Aggregation](#post-attack-aggregation))

### Grid Benchmark

**`defense_grid.py`** runs a defense with pre-defined configs, then evaluates and attacks the defended model.

```bash
uv run python scripts/whitebox/defense_grid.py meta-llama/Llama-3.1-8B-Instruct \
    --defense crl \
    --config-name base \
    --model-alias llama3_8b \
    --results-dir results/defense_grid
```

Run `uv run python scripts/whitebox/defense_grid.py --help` for all arguments.

```text
# Output structure:
results/defense_grid/
└── llama3_8b/
    └── crl/
        └── grid/
            └── trial_0/
                ├── defended_model/                    # (deleted unless --keep-checkpoints)
                ├── defense_eval/
                │   └── tamperbench_evaluation/
                │       ├── strong_reject/
                │       └── mmlu_pro_val/
                ├── post_attack/
                │   └── lora_finetune/
                │       └── base/
                │           └── tamperbench_evaluation/
                │               ├── strong_reject/
                │               └── mmlu_pro_val/
                └── trial_results.json                 # all metrics with prefixed keys
```

### Optuna Sweep

**`defense_sweep.py`** sweeps defense hyperparameters to find the configuration that best resists attacks. It loads the base config from `grid.yaml` and the search space from `single_objective_sweep.yaml`.

```bash
uv run python scripts/whitebox/defense_sweep.py meta-llama/Llama-3.1-8B-Instruct \
    --defense crl \
    --n-trials 20 \
    --model-alias llama3_8b \
    --results-dir results/defense_sweep
```

Each trial: suggests defense hyperparameters -> runs defense -> evaluates defended checkpoint -> runs attacks -> evaluates post-attack checkpoints -> aggregates post-attack metrics -> returns objective to Optuna.

Run `uv run python scripts/whitebox/defense_sweep.py --help` for all arguments.

```text
# Output structure:
results/defense_sweep/
└── llama3_8b/
    └── crl/
        └── optuna_single/
            ├── study.db        # Optuna SQLite database
            ├── best.yaml       # Top defense configs
            ├── best.json       # Trial summaries
            └── trial_0/
                └── ...         # same per-trial structure as grid
```

### Attack Modes

Each attack spec in the `attacks` list uses either `grid` or `sweep` mode:

**Grid mode** (`mode: grid`): loads the attack's `configs/whitebox/attacks/<attack>/grid.yaml` and runs fixed configs. When `config_name` is set (e.g., `"base"`), only that config is run. When `config_name: null`, all configs in the file are run and metrics are aggregated across them.

**Sweep mode** (`mode: sweep`): runs an inner Optuna sweep that optimizes attack hyperparameters from the attacker's perspective against the defended model. The inner sweep:
  1. Loads the attack's `grid.yaml` base config and `single_objective_sweep.yaml` search space
  2. Creates an inner Optuna study at `trial_N/post_attack/<attack>/optuna_single/`
  3. Runs `n_trials` attack trials, each suggesting hparams -> attacking -> evaluating
  4. Returns the best attack's metrics (the trial that caused the most harm)

Inner sweep artifacts (study.db, best.yaml, best.json) persist alongside eval results, allowing analysis of which attack configurations were most effective.

### Metric Keys

All metrics use dot-separated prefixed keys in `trial_results.json` and Optuna `user_attrs`:

```text
defense.strong_reject_score                          # defense checkpoint eval
defense.mmlu_pro_accuracy                            # defense checkpoint eval
post_attack.lora_finetune.strong_reject_score        # post-attack eval
post_attack.lora_finetune.mmlu_pro_accuracy          # post-attack eval
```

The primary optimization objective is the aggregated post-attack safety metric from the first attack, using the defender's optimization direction.

## Post-Attack Aggregation

When an attack runs in grid mode with multiple configs (`config_name: null`), post-attack metrics must be aggregated into a single value per evaluation. The aggregation method is configurable per attack spec via the `aggregation` field. Available methods are defined in the `AttackAggregationMethod` enum in `src/tamperbench/whitebox/utils/names.py` and registered in `AGGREGATION_REGISTRY` in `src/tamperbench/whitebox/utils/benchmark/attack_aggregation.py`.

### Configuration

Aggregation is configured per-attack in the `attacks` list of `grid.yaml` or `single_objective_sweep.yaml`:

```yaml
attacks:
  # Default: worst case across all configs (no aggregation block needed)
  - name: lora_finetune
    mode: grid
    config_name: null

  # Top-N average: mean of the 3 worst configs
  - name: lora_finetune
    mode: grid
    config_name: null
    aggregation:
      method: top_n_average
      n: 3

  # Bounded worst case: worst case among configs that kept MMLU
  # above 80% of the defended model's score
  - name: full_parameter_finetune
    mode: grid
    config_name: null
    aggregation:
      method: bounded_worst_case
      bound_eval: mmlu_pro_val
      bound_min_retention: 0.8
```

**`worst_case`** (default): selects the single config that maximizes attacker harm, per eval. This is the most conservative measure of defense robustness.

**`top_n_average`**: sorts configs by attacker harm, averages the top N. Smooths out outlier attack configs. Controlled by `n` (default: 3).

**`bounded_worst_case`**: compares each config's post-attack score on `bound_eval` against the defended model's score on that same eval. Only configs where the post-attack score is >= `bound_min_retention * defense_score` are considered. This filters out attack configs that "won" by destroying model capability entirely. Returns `NaN` if no configs pass the bound. Both `bound_eval` and `bound_min_retention` are required when using this method (validated at config load time by `AttackAggregationConfig` in `defense_config.py`).

When `config_name` points to a single config (e.g., `"base"`), aggregation has no effect since there is only one result.

## Capability Guards

Capability guards allow defense sweeps to skip the expensive post-defense attack phase when a defense trial degrades model utility too much. This saves compute by avoiding attack runs on defenses that are already too destructive.

Configured in `single_objective_sweep.yaml` under `capability_guards`:

```yaml
capability_guards:
    mmlu_pro_val:
        min_retention: 0.8  # defended model must keep >= 80% of base model's score
```

How it works:

1. At sweep start, the base (undefended) model is evaluated once on all guarded metrics to establish baselines. Results are saved to `<sweep_results_dir>/capability_baselines/`.
2. After each trial's defense step, the defended model's scores are compared against `base_score * min_retention`.
3. If any guarded metric drops below the threshold, the trial skips the attack phase and receives a penalty value so Optuna learns to avoid that hyperparameter region.

Guarded metrics must be listed in `defense_evals` -- this is validated at config load time.

## Tips

1. **Always use `--model-alias`** -- keeps results organized and enables Optuna resume
2. **Defense checkpoints are cleaned up** -- only eval artifacts and `trial_results.json` persist; model weights are deleted after all attacks complete to save disk space. Use `--keep-checkpoints` to retain them for debugging or when the sweep is small.
3. **Inner attack sweeps** -- use `mode: sweep` in the attacks config to run an inner Optuna optimization from the attacker's perspective against each defended model
4. **Aggregation defaults to worst case** -- if you don't specify an `aggregation` block, the pipeline uses worst-case (max attacker harm) across all attack configs
