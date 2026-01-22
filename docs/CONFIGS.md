# Configuration Reference

This guide covers how configurations are structured and how to customize them.

These configs are primarily designed to support the benchmark scripts (`benchmark_grid.py` and `optuna_single.py`) for exhaustive benchmarking. They map directly to the dataclass configs used by each attack/evaluation class.

<!-- TODO: Consider migrating to Hydra for more flexible config composition -->

## Config File Location

Default attack configurations live in `configs/whitebox/attacks/`. This directory can be copied and customized as needed—use `--configs-dir` to point scripts to your custom configs.

```text
configs/whitebox/attacks/<attack_name>/
├── grid.yaml                    # Required for benchmark_grid.py
└── single_objective_sweep.yaml  # Required for optuna_single.py
```

**Minimum requirements for scripts:**

- `grid.yaml` with at least a `base` key defining default parameters
- `single_objective_sweep.yaml` if using `optuna_single.py` for sweeps
- Attack subdirectory names must match `AttackName` enum values

## Naming Conventions

Config directory names must match `AttackName` enum values exactly. Evaluation names in `evals:` lists must match `EvalName` enum values. See `src/safetunebed/whitebox/utils/names.py` for all valid names.

## grid.yaml Structure

Used by `benchmark_grid.py` for static benchmark runs.

```yaml
base: &base_cfg
    model_config:
        template: plain
        max_generation_length: 1024
        inference_batch_size: 16
    evals: [strong_reject, mmlu_pro_val]
    per_device_train_batch_size: 8
    learning_rate: 0.0001
    num_train_epochs: 1
    lr_scheduler_type: constant
    optim: adamw_torch
    lora_rank: 16
    max_steps: -1

# Additional variants inherit from base
high_lr:
    <<: *base_cfg
    learning_rate: 0.001

low_rank:
    <<: *base_cfg
    lora_rank: 8
```

### Config to Dataclass Mapping

YAML fields map directly to the attack's config dataclass. For example, with `LoraFinetuneConfig`:

```python
# LoraFinetuneConfig dataclass fields:
@dataclass
class LoraFinetuneConfig(FullParameterFinetuneConfig):
    lora_rank: int
    # Inherits: learning_rate, per_device_train_batch_size, max_steps, ...
```

```yaml
# grid.yaml - fields map 1:1 to dataclass attributes
base:
    lora_rank: 16
    learning_rate: 0.0001
    per_device_train_batch_size: 8
    # ...
```

See individual config dataclasses in `src/safetunebed/whitebox/attacks/<attack>/` for all available parameters.

### Evaluations

The `evals` field specifies which evaluations to run after the attack completes:

```yaml
evals: [strong_reject, mmlu_pro_val]
```

Values must match `EvalName` enum (see Evaluation Name Mapping above). Multiple evals can be specified.

## single_objective_sweep.yaml Structure

Used by `optuna_single.py` for hyperparameter optimization.

```yaml
evals: [strong_reject, mmlu_pro_val]
sweep:
  per_device_train_batch_size:
    choices: [8, 16, 32, 64]

  learning_rate:
    type: "float"
    low: 1.0e-6
    high: 1.0e-3
    log: true

  max_steps:
    choices: [16, 64, 128, 256, 512]

  lr_scheduler_type:
    choices: [constant, cosine]

  model_config.template:
    choices: [plain, instruction_response, generic_chat]

  lora_rank:
    type: categorical
    choices: [8, 16, 32, 64]
```

### Parameter Types

**Categorical (choices):**

```yaml
lr_scheduler_type:
  choices: [constant, cosine, linear]
```

**Float range:**

```yaml
learning_rate:
  type: "float"
  low: 1.0e-6
  high: 1.0e-3
  log: true    # Log-uniform sampling
```

**Integer range:**

```yaml
num_layers:
  type: "int"
  low: 1
  high: 12
  step: 1      # Optional step size
```

**Explicit categorical:**

```yaml
lora_rank:
  type: categorical
  choices: [8, 16, 32, 64]
```

### Nested Parameters

Use dotted notation for nested config values:

```yaml
sweep:
  model_config.template:
    choices: [plain, generic_chat]

  model_config.max_generation_length:
    choices: [512, 1024, 2048]
```

## Model Config Templates

The `template` field in `model_config` controls prompt formatting.

 | Template               | Description                                    |
 | ---------------------- | ---------------------------------------------- |
 | `plain`                | No special formatting                          |
 | `generic_chat`         | Standard chat format with user/assistant turns |
 | `instruction_response` | Instruction/response pairs                     |

Templates are defined in `src/safetunebed/whitebox/utils/models/templates.py`.

### Custom Templates

Add new templates by extending the template registry:

```python
# In templates.py
TEMPLATES["my_template"] = {
    "user_prefix": "<|user|>\n",
    "assistant_prefix": "<|assistant|>\n",
    "end_turn": "<|end|>",
}
```

## When to Edit Which File

 | Goal                               | File to Edit                  |
 | ---------------------------------- | ----------------------------- |
 | Run benchmarks with fixed params   | `grid.yaml`                   |
 | Add new config variant             | `grid.yaml` (new key)         |
 | Change hyperparameter search space | `single_objective_sweep.yaml` |
 | New attack type                    | Create new directory          |

## Creating a New Attack Config

1. Create directory: `configs/whitebox/attacks/<attack_name>/`

2. Add `grid.yaml` with base config:

   ```yaml
   base:
       model_config:
           template: plain
           max_generation_length: 1024
           inference_batch_size: 16
       evals: [strong_reject, mmlu_pro_val]
       # Add attack-specific params
   ```

3. Add `single_objective_sweep.yaml` (if using Optuna sweeps):

   ```yaml
   evals: [strong_reject, mmlu_pro_val]
   sweep:
     # Define search space
   ```

4. Ensure attack subdirectory name matches `AttackName` enum in `names.py`

## Next Steps

- [USAGE.md](USAGE.md) - Running benchmarks with these configs
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Adding new attacks (requires config)
