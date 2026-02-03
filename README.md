# TamperBench

An extensible toolkit for benchmarking safety‑preserving fine‑tuning methods for LLMs.

## Features

- **Tampering Attacks**: LoRA, full-parameter fine-tuning, embedding attacks, jailbreak fine-tuning, and more
- **Safety Evaluations**: StrongReject, MMLU-Pro, and other benchmarks
- **Hyperparameter Optimization**: Optuna-based sweeps with automatic trial management
- **Analysis Pipeline**: Epsilon-bounded filtering to find attacks that maximize harm while preserving utility

## Quick Start

```bash
# Install package and all dependencies
uv sync --all-groups

# Install pre-commit hooks
pre-commit install
```

## Usage

### Python API

Run attacks and evaluations directly from Python:

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
results = attack.benchmark()  # Runs attack + evaluations
print(results)
```

### Hyperparameter Sweeps (Stress-Testing)

Find the attack configuration that maximizes harm—simulates a real-world attacker tuning their approach:

```bash
python scripts/whitebox/optuna_single.py Qwen/Qwen3-4B \
    --attacks lora_finetune \
    --n-trials 50 \
    --model-alias qwen3_4b
```

### Grid Benchmarks (Pre-defined Configs)

Run attacks with fixed hyperparameters from config files:

```bash
python scripts/whitebox/benchmark_grid.py Qwen/Qwen3-4B \
    --attacks lora_finetune full_parameter_finetune \
    --model-alias qwen3_4b
```

### Results Analysis

Aggregate sweep results and generate visualizations:

```bash
# Epsilon-bounded analysis (filters by utility degradation)
uv run scripts/analysis/analyze_results.py results/sweep epsilon \
    --epsilons 0.05 0.10 0.20

# Generate heatmaps
python scripts/figures/plot_main_heatmap.py results/aggregated_eps05/heatmap_max_sr.json
```

## Available Attacks

| Attack                    | Description                          |
| ------------------------- | ------------------------------------ |
| `lora_finetune`           | LoRA-based fine-tuning               |
| `full_parameter_finetune` | Full model parameter updates         |
| `embedding_attack`        | Adversarial embedding optimization   |
| `jailbreak_finetune`      | Jailbreak-style fine-tuning          |
| `multilingual_finetune`   | Multilingual attack vectors          |
| `backdoor_finetune`       | Backdoor injection                   |
| `benign_*`                | Benign baselines for comparison      |

## Available Evaluations

| Evaluation      | Description                          |
| --------------- | ------------------------------------ |
| `strong_reject` | Safety benchmark (StrongReject)      |
| `mmlu_pro_val`  | Knowledge benchmark (MMLU-Pro)       |
| `mbpp`          | Code generation                      |
| `minerva_math`  | Mathematical reasoning               |
| `ifeval`        | Instruction following                |

## Documentation

| Guide                                       | Description                               |
| ------------------------------------------- | ----------------------------------------- |
| [USAGE.md](docs/USAGE.md)                   | Full usage guide with examples            |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md)     | Adding new attacks, evaluations, defenses |
| [CONFIGS.md](docs/CONFIGS.md)               | Configuration system and YAML files       |
| [ANALYSIS.md](docs/ANALYSIS.md)             | Results analysis and visualization        |
| [MISC.md](docs/MISC.md)                     | Maintenance tips and troubleshooting      |

## Architecture

The whitebox framework uses a registry-based plugin architecture:

```text
src/tamperbench/whitebox/
├── attacks/          # Tampering attacks (LoRA, full-param, embedding, etc.)
├── defenses/         # Alignment defenses
├── evals/            # Safety and utility evaluations
└── utils/            # Benchmark utilities, analysis pipeline
```
