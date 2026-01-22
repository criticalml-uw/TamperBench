# Results Analysis

This guide covers how to analyze and visualize benchmark results.

## Overview

After running Optuna sweeps, you can:

1. Run the analysis pipeline to aggregate results across models/attacks
2. Generate heatmaps and visualizations
3. Manually inspect Optuna databases

The analysis pipeline is **optional** - you can always load Optuna studies directly.

**Prerequisites:**

- Results directory must follow the expected structure (see [Output Structure](#output-structure) below)
- Analysis config must list the models and attacks to process (see [Analysis Config](#analysis-config-optional))

## Running the Analysis Pipeline

Use `scripts/analysis/analyze_results.py` to aggregate sweep results.

### Epsilon-Bounded Analysis

Filters trials by utility degradation (MMLU drop) and selects the best attack within constraints:

```bash
# Basic epsilon-bounded analysis
uv run scripts/analysis/analyze_results.py results/sweep_results epsilon \
    --epsilons 0.05 0.20 0.50

# With custom config
uv run scripts/analysis/analyze_results.py results/sweep_results epsilon \
    --epsilons 0.05 \
    --config configs/analysis/my_config.yaml \
    --n-trials 40
```

Run `python scripts/analysis/analyze_results.py epsilon --help` for all available arguments.

## Output Structure

The pipeline creates `aggregated_epsXX/` directories:

```text
results/sweep_results/
├── model_a/
│   └── lora_finetune/
│       └── optuna_single/
│           └── study.db
├── model_b/
│   └── ...
├── aggregated_eps05/              # Created by analysis
│   ├── heatmap_max_sr.json        # Data for plotting
│   ├── model_a/
│   │   └── lora_finetune/
│   │       ├── trial_42_config.yaml
│   │       └── trial_42_eval/
│   └── model_b/
│       └── ...
├── aggregated_eps20/
└── aggregated_eps50/
```

### heatmap_max_sr.json

Contains matrices for visualization:

```json
{
  "safety_raw": [[0.12, 0.45, ...], ...],
  "safety_delta": [[0.0, 0.33, ...], ...],
  "utility_raw": [[0.65, 0.62, ...], ...],
  "utility_delta": [[0.0, -0.03, ...], ...],
  "models": ["model_a", "model_b", ...],
  "attacks": ["no_weight_modification", "lora_finetune", ...],
  "metadata": {
    "method": "epsilon_bounded",
    "epsilon": 0.05,
    "n_trials": 40
  }
}
```

## Analysis Config (Optional)

The analysis config lets you customize pipeline behavior, but the defaults are fine for most users. The pipeline provides:

- **Epsilon-bounded filtering**: Select best attacks that stay within a utility degradation threshold
- **Trial filtering**: Filter and rank trials across multiple models/attacks
- **Best hyperparameters extraction**: Automatically output `grid.yaml` files with winning configs
- **Visualization data**: Generate structured JSON for heatmaps and comparison plots

The config at `configs/analysis/default.yaml` controls how results are filtered, grouped, and visualized.

### Key Sections

```yaml
attacks:
  preferred_order:
    - no_weight_modification
    - lora_finetune
    - full_parameter_finetune
  categories:
    stealthy:
      display_name: "Stealthy"
      attacks: [backdoor_finetune, style_modulation_finetune]

models:
  preferred_order:
    - llama3_1b_base
    - llama3_1b_instruct
  include_models:
    - llama3_1b_base
    - llama3_1b_instruct

epsilon_bounded:
  baseline_attack: no_weight_modification
  utility_metric: mmlu_pro_val
  safety_metric: strong_reject
  default_epsilons: [0.01, 0.03, 0.05]
  default_n_trials: 20
```

The config dataclass is defined in `src/safetunebed/whitebox/utils/analysis/config.py`.

## Visualization Scripts

Generate heatmaps from the JSON output:

### Main Heatmap

```bash
python scripts/figures/plot_main_heatmap.py \
    results/aggregated_eps05/heatmap_max_sr.json

# Custom output path
python scripts/figures/plot_main_heatmap.py \
    results/aggregated_eps05/heatmap_max_sr.json \
    -o figures/main_heatmap.png
```

Shows StrongReject scores grouped by attack category with averages.

### Supplemental Heatmap

```bash
python scripts/figures/plot_supplemental_heatmap.py \
    results/aggregated_eps05/heatmap_max_sr.json
```

Shows StrongReject + MMLU delta for each attack (2 rows per attack).

## Manual Inspection

### Loading Optuna Studies

```python
import optuna

# Load a study directly
study = optuna.load_study(
    study_name="lora_finetune_single",
    storage="sqlite:///results/model/lora_finetune/optuna_single/study.db"
)

# Best trial
print(f"Best value: {study.best_trial.value}")
print(f"Best params: {study.best_trial.params}")

# All trials
for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print(f"Trial {trial.number}: {trial.value}")
```

### Trial User Attributes

Each trial stores metadata in `user_attrs`:

```python
trial = study.best_trial

# Merged config used for this trial
config = trial.user_attrs["merged_config"]

# Evaluation metrics
metrics = trial.user_attrs["eval_metrics"]
print(f"StrongReject: {metrics['strong_reject']}")
print(f"MMLU: {metrics['mmlu_pro_val']}")
```

### Evaluation Results

Raw evaluation outputs are in each trial's directory:

```text
trial_42/
└── safetunebed_evaluation/
    ├── strong_reject/
    │   ├── inferences.parquet    # Model outputs
    │   ├── scores.parquet        # Per-sample scores
    │   └── results.parquet       # Aggregate metrics
    └── mmlu_pro_val/
        └── ...
```

Load with pandas:

```python
import pandas as pd

results = pd.read_parquet(
    "results/.../trial_42/safetunebed_evaluation/strong_reject/results.parquet"
)
print(results)
```

## Workflow Example

Complete analysis workflow:

```bash
# 1. Run sweeps for multiple models
for model in llama3_1b qwen3_0_6b; do
    python scripts/whitebox/optuna_single.py "path/to/$model" \
        --attacks lora_finetune full_parameter_finetune \
        --n-trials 50 \
        --model-alias "$model" \
        --results-dir results/my_sweep
done

# 2. Run epsilon-bounded analysis
uv run scripts/analysis/analyze_results.py results/my_sweep epsilon \
    --epsilons 0.05 0.10 0.20

# 3. Generate visualizations
python scripts/figures/plot_main_heatmap.py \
    results/my_sweep/aggregated_eps05/heatmap_max_sr.json \
    -o figures/main_results.png

python scripts/figures/plot_supplemental_heatmap.py \
    results/my_sweep/aggregated_eps05/heatmap_max_sr.json \
    -o figures/supplemental.png
```

## Reusing Hyperparameters Across Models

A key use-case is transferring the best hyperparameters found on one model to another. The analysis pipeline outputs `grid.yaml` files in each model's aggregated directory that can be used directly with `benchmark_grid.py`.

### Use Case: Transfer from Smaller to Larger Model

Run a full sweep on a smaller, faster model, then apply the best configs to a larger model:

```bash
# 1. Sweep on qwen3_8b (faster)
python scripts/whitebox/optuna_single.py Qwen/Qwen3-8B \
    --attacks lora_finetune full_parameter_finetune \
    --n-trials 50 \
    --model-alias qwen3_8b \
    --results-dir results/sweep

# 2. Run analysis to get best configs
uv run scripts/analysis/analyze_results.py results/sweep epsilon \
    --epsilons 0.05

# 3. Apply best configs to qwen3_32b (larger)
python scripts/whitebox/benchmark_grid.py Qwen/Qwen3-32B \
    --attacks lora_finetune full_parameter_finetune \
    --model-alias qwen3_32b \
    --configs-dir results/sweep/aggregated_eps05/qwen3_8b/
```

### Output Structure After Analysis

The aggregated directory contains `grid.yaml` files ready for reuse:

```text
results/sweep/aggregated_eps05/
└── qwen3_8b/
    ├── lora_finetune/
    │   ├── grid.yaml                  # Best config in grid format
    │   ├── trial_42_config.yaml       # Original trial config
    │   └── trial_42_eval/             # Copied evaluation results
    └── full_parameter_finetune/
        ├── grid.yaml
        └── ...
```

The `grid.yaml` contains the hyperparameters from the best trial:

```yaml
qwen3_8b_params:
    learning_rate: 0.0001
    lora_rank: 16
    max_steps: 256
    per_device_train_batch_size: 8
    # ... other params
```

## Next Steps

- [USAGE.md](USAGE.md) - Running the initial sweeps
- [CONFIGS.md](CONFIGS.md) - Configuring sweep parameters
