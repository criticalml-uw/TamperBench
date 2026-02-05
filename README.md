<div align="center">

# TamperBench

**Benchmark the tamper-resistance of open-weight large language models (LLMs)**

[![Transformers](https://img.shields.io/badge/transformers-%E2%89%A54.49-orange?style=flat-square&logo=huggingface&logoColor=white)](https://github.com/huggingface/transformers)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet?style=flat-square&logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000?style=flat-square)](https://github.com/astral-sh/ruff)
[![Basedpyright](https://img.shields.io/badge/type%20checked-basedpyright-blue?style=flat-square)](https://github.com/DetachHead/basedpyright)

</div>

---

- :crossed_swords: Red-team LLMs with tampering attacks like fine-tuning, jailbreak-tuning, embedding attacks, and others.
- :shield: Evaluate safety and utility with StrongREJECT, MMLU-Pro, and others.
- :gear: Run rigorous hyperparameter sweeps to stress-test models under worst-case attacker configurations.
- :snake: Use a typed Python API to run attacks and evaluations programmatically.
- :jigsaw: Add or use custom attacks and evaluations with an extensible, decorator-based plugin architecture.

<br>

<div align="center">

[![TamperBench Toolkit](assets/tamperbench_toolkit.png)](assets/tamperbench_toolkit.png)

</div>

## :rocket: Getting Started

### Installation

```bash
git clone https://github.com/criticalml-uw/tamperbench.git
cd tamperbench
uv sync --all-groups
pre-commit install
```

### :zap: Run a Benchmark

Stress-test Qwen3-4B with 50 hyperparameter trials using the `lora_finetune` attack:

```bash
uv run scripts/whitebox/optuna_single.py Qwen/Qwen3-4B \
    --attacks lora_finetune \
    --n-trials 50
```

Run Qwen3-4B against a fixed grid of hyperparameters for multiple attacks (see [CONFIGS.md](docs/CONFIGS.md) for available configurations):

```bash
uv run scripts/whitebox/benchmark_grid.py Qwen/Qwen3-4B \
    --attacks lora_finetune full_parameter_finetune
```

### :snake: Python API

Configure and run a LoRA fine-tuning attack against Llama-3.1-8B-Instruct, then evaluate safety (StrongReject) and utility (MMLU-Pro) on the tampered model:

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
results = attack.benchmark()
print(results)
```

## :bar_chart: Results

Benchmarking tamper resistance across 21 open-weight LLMs. For each model-attack pair, we select the configuration that maximizes harmfulness (StrongREJECT score) while constraining utility loss to at most 10% MMLU-Pro drop. Darker cells indicate higher harmfulness; lighter cells indicate greater tamper resistance.

<div align="center">

[![StrongReject Results Heatmap](assets/tamperbench_results_heatmap.png)](assets/tamperbench_results_heatmap.png)

</div>

## :link: Quick Links

| [Usage Guide](docs/USAGE.md) | [Contributing](docs/CONTRIBUTING.md) | [Configs](docs/CONFIGS.md) | [Analysis](docs/ANALYSIS.md) |
| --- | --- | --- | --- |
| Full usage guide, Python API, and CLI examples | Adding new attacks, evaluations, and defenses | Configuration system and YAML files | Results analysis, epsilon-bounded filtering, and visualization |
