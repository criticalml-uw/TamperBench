# Usage Guide

This guide covers how to run safety benchmarks using the whitebox framework.

## Use Cases

 | Use Case                             | Approach                 | When to Use                              |
 | ------------------------------------ | ------------------------ | ---------------------------------------- |
 | Run a single attack on a model       | Direct Python API        | Testing, debugging, custom workflows     |
 | Run a single evaluation              | Direct Python API        | Evaluating existing checkpoints          |
 | Apply a defense to a model           | Direct Python API        | Alignment research                       |
 | Run attacks with pre-defined configs | `benchmark_grid.py`      | Fixed hyperparameters, quick tests       |
 | Sweep hyperparameters (stress-test)  | `optuna_single.py`       | Research, modeling real-world attacker   |
 | Benchmark multiple models            | Shell scripts with loops | Large-scale experiments                  |

## Direct Python API

### Running an Attack

```python
from tamperbench.whitebox.attacks.lora_finetune.lora_finetune import (
    LoraFinetune,
    LoraFinetuneConfig,
)
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import EvalName

config = LoraFinetuneConfig(
    input_checkpoint_path="meta-llama/Llama-3.1-8B-Instruct",
    out_dir="results/my_attack",
    evals=[EvalName.STRONG_REJECT, EvalName.MMLU_PRO_VAL],
    model_config=ModelConfig(
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        end_turn="<|eot_id|>\n",
        max_generation_length=1024,
        inference_batch_size=16,
    ),
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    num_train_epochs=1,
    max_steps=-1,
    lr_scheduler_type="constant",
    optim="adamw_torch",
    lora_rank=16,
    random_seed=42,
)

attack = LoraFinetune(attack_config=config)
results = attack.benchmark()  # Runs attack + all evals, returns DataFrame[EvaluationSchema]

# Results DataFrame has columns: metric_name, metric_value
print(results)

# Output structure:
# results/my_attack/
# ├── checkpoint/                    <- Fine-tuned model weights
# └── tamperbench_evaluation/
#     ├── strong_reject/
#     │   ├── inferences.parquet     <- Model generations (prompt, response)
#     │   ├── scores.parquet         <- Per-sample scores (prompt, response, score)
#     │   └── results.parquet        <- Aggregate metrics (metric_name, metric_value)
#     └── mmlu_pro_val/
#         └── ...

# Loading results from a completed evaluation:
strong_reject_eval = attack.evals[EvalName.STRONG_REJECT]
print(f"StrongReject score: {strong_reject_eval.load_result_objective()}")
inferences = strong_reject_eval.load_inferences()   # DataFrame[InferenceSchema]: prompt, response
scores = strong_reject_eval.load_scores()           # DataFrame[ScoreSchema]: prompt, response, score

# Alternatively, parquet files can be read directly via pd.read_parquet()
```

### Running an Evaluation

```python
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
)
from tamperbench.whitebox.utils.models.config import ModelConfig

config = StrongRejectEvaluationConfig(
    checkpoint_path="results/my_attack/checkpoint",
    out_dir="results/my_attack/eval",
    model_config=ModelConfig(
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        end_turn="<|eot_id|>\n",
        max_generation_length=1024,
        inference_batch_size=16,
    ),
)

evaluation = StrongRejectEvaluation(config)
results = evaluation.run_evaluation()  # Returns DataFrame[EvaluationSchema]

# Get the objective metric value directly
print(f"StrongReject score: {evaluation.load_result_objective()}")
```

### Applying a Defense

See [DEFENSES.md](DEFENSES.md) for the full defense guide, including grid benchmarks and Optuna sweeps.

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

## Available Attacks and Evaluations

See `AttackName` and `EvalName` enums in `src/tamperbench/whitebox/utils/names.py` for all available attacks and evaluations. See [CONTRIBUTING.md](../CONTRIBUTING.md) for adding custom attacks.

## Benchmark Scripts

The scripts below are orchestrators that use the direct Python API internally. They handle configuration loading, result organization, and hyperparameter optimization while calling the same attack/evaluation classes shown above.

### Quick Start: Optuna Sweep (Stress-Testing)

**`optuna_single.py`** sweeps hyperparameters to find the configuration that maximizes attack success. This simulates a real-world attacker who tunes their approach—the most rigorous way to stress-test model safety.

```bash
python scripts/whitebox/optuna_single.py Qwen/Qwen3-4B \
    --attacks lora_finetune \
    --n-trials 50 \
    --model-alias qwen3_4b
```

This runs:

- Loads sweep search space from `configs/whitebox/attacks/lora_finetune/single_objective_sweep.yaml`
- Runs 50 trials with different hyperparameter combinations
- Each trial: trains attack → evaluates on StrongReject + MMLU-Pro → records metrics
- Optuna optimizes to maximize StrongReject score (attack success)

```text
# Output structure:
results/optuna_single_2024_01_15/
└── qwen3_4b/
    └── lora_finetune/
        └── optuna_single/
            ├── study.db           <- Optuna SQLite database (all trial data)
            ├── best.yaml          <- Top 3 configurations as grid.yaml format
            ├── best.json          <- Trial metadata for top configs
            ├── trial_0/
            │   ├── checkpoint/    <- Model weights
            │   └── tamperbench_evaluation/
            │       ├── strong_reject/
            │       │   ├── inferences.parquet
            │       │   ├── scores.parquet
            │       │   └── results.parquet
            │       └── mmlu_pro_val/
            ├── trial_1/
            └── ...
```

Reading results:

```python
import optuna

study = optuna.load_study(
    study_name="lora_finetune_single",
    storage="sqlite:///results/.../study.db"
)
print(f"Best StrongReject: {study.best_value}")
print(f"Best params: {study.best_params}")
```

Run `python scripts/whitebox/optuna_single.py --help` for all available arguments.

**Using `--model-alias` is recommended.** It organizes all attacks for a model under one directory, enables resuming interrupted sweeps (same alias continues from last trial), and keeps results clean for analysis.

### Quick Start: Grid Benchmark (Pre-defined Configs)

**`benchmark_grid.py`** runs attacks with pre-defined hyperparameter sets from `grid.yaml`. Use this when you already know the configs you want to test.

```bash
python scripts/whitebox/benchmark_grid.py Qwen/Qwen3-4B \
    --attacks lora_finetune full_parameter_finetune \
    --model-alias qwen3_4b
```

This runs:

- Loads configurations from `configs/whitebox/attacks/<attack>/grid.yaml`
- Each config variant in grid.yaml is run as a separate benchmark
- Trains attack → evaluates → saves results

```text
# Output structure:
results/grid_2024_01_15/
└── qwen3_4b/
    └── lora_finetune/
        └── base/                  <- Config variant name from grid.yaml
            ├── checkpoint/
            └── tamperbench_evaluation/
                ├── strong_reject/
                │   ├── inferences.parquet
                │   ├── scores.parquet
                │   └── results.parquet
                └── mmlu_pro_val/
```

Run `python scripts/whitebox/benchmark_grid.py --help` for all available arguments.

**Using `--model-alias` is recommended.** It organizes all attacks for a model under one directory, enabling parallel execution of different attacks and clean structure for analysis.

After running analysis (see [ANALYSIS.md](ANALYSIS.md)), `aggregated_epsXX/` directories contain `grid.yaml` files with best configs. Use these on new models via `--configs-dir`.

## Example Scripts

**Example Single Objective Sweep on SLURM**:

```bash
#!/bin/bash
#SBATCH --job-name=optuna_single
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

MODEL="Qwen/Qwen3-4B"
MODEL_ALIAS="qwen3_4b"
RESULTS_DIR="results/nov7_trial"
CONFIGS_DIR="configs/whitebox/attacks_qwen"

uv run scripts/whitebox/optuna_single.py "$MODEL" \
    --attacks lora_finetune \
    --n-trials 20 \
    --results-dir "$RESULTS_DIR" \
    --model-alias "$MODEL_ALIAS" \
    --configs-dir "$CONFIGS_DIR"
```

**Example Grid Sweep on SLURM**:

```bash
#!/bin/bash
#SBATCH --job-name=benchmark_grid
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

uv run scripts/whitebox/benchmark_grid.py Qwen/Qwen3-32B \
    --attacks lora_finetune embedding_attack \
    --model-alias qwen3_32b \
    --configs-dir results/aggregated_eps05/qwen3_8b/
```

## Multiple Models

```bash
#!/bin/bash
MODELS=("Qwen/Qwen3-4B" "Qwen/Qwen3-8B")
ALIASES=("qwen3_4b" "qwen3_8b")

for i in "${!MODELS[@]}"; do
    python scripts/whitebox/optuna_single.py "${MODELS[$i]}" \
        --attacks lora_finetune \
        --n-trials 50 \
        --model-alias "${ALIASES[$i]}" \
        --results-dir results/sweep
done
```

## Tips

1. **Always use `--model-alias`** - Keeps results organized and enables resume
2. **Use `--configs-dir`** - Point to custom configs or reuse from sweeps
3. **Check disk space** - Checkpoints accumulate; see [MISC.md](MISC.md) for cleanup
4. **Optuna for research** - Sweeps find the strongest attacker configuration
5. **Grid for quick tests** - Use pre-defined configs when you know what to run

## Next Steps

- [DEFENSES.md](DEFENSES.md) - Benchmarking and sweeping alignment defenses
- [CONFIGS.md](CONFIGS.md) - How to edit and create configuration files
- [ANALYSIS.md](ANALYSIS.md) - Analyzing and visualizing results
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Adding new attacks and evaluations
