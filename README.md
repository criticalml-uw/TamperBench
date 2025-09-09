# SafeTuneBed

An experimental toolkit for exploring safety-preserving fine-tuning of large language models.

> **Heads‑up:** this guide will migrate to a dedicated `CONTRIBUTING.md` later.
> Until then, treat it as the source of truth for contributing.

> **Hardware:** most scripts assume access to a GPU (currently tuned for NVIDIA A100 80 GB). Check `torch.cuda.is_available()` before launching heavy jobs.

## Getting Started

Clone the repository and install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/your-org/SafeTuneBed.git
cd SafeTuneBed
uv sync
```

Install pre‑commit hooks so linting and formatting run automatically:

```bash
uv tool run pre-commit install
```

Run a quick sanity test to verify your environment:

```bash
uv run tests/evals/test_strong_reject.py
```

## Benchmarking *(experimental)*

The benchmarking API may change. A tentative workflow:

```bash
uv run scripts/whitebox/benchmark_grid.py /path/to/model \
    --attacks lora_finetune embedding_attack
```

Configuration grids live under `configs/whitebox/attacks/`. Results default to `results/`.

## Contribution Overview

When proposing a contribution, clearly describe the **optimization direction** (what the component aims to maximize or minimize) for attacks, defenses, and evaluations.

### Infrastructure
- Core abstractions such as plugin systems, interfaces, or utilities that reduce boilerplate.
- Should make it easy for third parties to plug in new attacks, defenses, or evaluations.

### Attack
- Methods that tamper with model weights or embeddings (e.g., jailbreak tuning, latent perturbations).
- Usually strive to **maximize** a harmful metric.

### Evaluation
- Metrics or benchmarks that quantify refusal, harmful knowledge, benign capability retention, or instruction following.
- Examples include MMLU, MTBench, and StrongReject.

### Defense *(work in progress)*
- Techniques like TAR, Booster, or RepNoise that mitigate tampering.
- Currently most defenses rely on HuggingFace checkpoints.

### Dataset Guidelines
- **Do not commit datasets** to the repository.
- Upload any new dataset to the HuggingFace Hub and ensure it is **public**.
- Include a dataset card with licensing information and cite all sources.
- Reference datasets by their HuggingFace identifier in configs or code.

### Developer Environment
- Recommended editor: VS Code with the **Ruff** and **BasedPyright** extensions.
- Suppress third‑party Pyright errors by adding `# pyright: ignore` headers or extending the `pyproject.toml` ignore list.

## Quick Guides

### Adding an Attack
1. Create a package under `src/safetunebed/whitebox/attacks/<your_attack>/`.
2. Define a config dataclass inheriting from `TamperAttackConfig`.
3. Implement a subclass of `TamperAttack` with a unique `AttackName`.
4. Implement `run_attack` to produce the tampered checkpoint and `evaluate` to invoke the desired evaluations.
5. Register the name in `src/safetunebed/whitebox/utils/names.py` and in `ATTACKS_MAP` within `scripts/whitebox/benchmark_grid.py`.
6. Provide a default grid at `configs/whitebox/attacks/<your_attack>/grid.yaml`.
7. Add tests under `tests/attacks/`.

**Example skeleton**

```python
# src/safetunebed/whitebox/attacks/my_attack/__init__.py
from dataclasses import dataclass
import polars as pl
from safetunebed.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from safetunebed.whitebox.utils.names import AttackName
from safetunebed.whitebox.evals.output_schema import EvaluationSchema

@dataclass
class MyAttackConfig(TamperAttackConfig):
    lr: float = 1e-3

class MyAttack(TamperAttack[MyAttackConfig]):
    name: AttackName = AttackName.MY_ATTACK

    def run_attack(self) -> None:
        # 1. Load model from self.attack_config.input_checkpoint_path
        # 2. Apply tampering / fine-tuning
        # 3. Save to self.output_checkpoint_path
        ...

    def evaluate(self) -> pl.DataFrame[EvaluationSchema]:
        # Instantiate evaluations from self.attack_config.evals
        # and concatenate their results
        ...
```

### Adding an Evaluation
1. Create a package under `src/safetunebed/whitebox/evals/<your_eval>/`.
2. Define a config dataclass inheriting from `WhiteBoxEvaluationConfig`.
3. Implement a subclass of `WhiteBoxEvaluation` and specify:
   - `name` (an `EvalName` entry)
   - `objective` (`MetricName` for hyper‑parameter search)
   - `attacker_direction` and `defender_direction` (instances of `OptimizationDirection`)
4. Implement the methods `compute_inferences`, `compute_scores`, and `compute_results`.
5. Expose the classes in `src/safetunebed/whitebox/evals/__init__.py` and register the name in `src/safetunebed/whitebox/utils/names.py`.
6. Add tests under `tests/evals/`.

**Example skeleton**

```python
# src/safetunebed/whitebox/evals/my_eval/__init__.py
from dataclasses import dataclass
import polars as pl
from datasets import load_dataset
from safetunebed.whitebox.evals.base import (
    WhiteBoxEvaluation, WhiteBoxEvaluationConfig,
)
from safetunebed.whitebox.evals.output_schema import (
    EvaluationSchema, InferenceSchema, ScoreSchema,
)
from safetunebed.whitebox.utils.names import (
    EvalName, MetricName, OptimizationDirection,
)

@dataclass
class MyEvalConfig(WhiteBoxEvaluationConfig):
    dataset_name: str = "username/my_dataset"

class MyEval(WhiteBoxEvaluation[MyEvalConfig]):
    name = EvalName.MY_EVAL
    objective = MetricName.MY_EVAL_SCORE
    attacker_direction = OptimizationDirection.MAXIMIZE
    defender_direction = OptimizationDirection.MINIMIZE

    def compute_inferences(self) -> pl.DataFrame[InferenceSchema]:
        data = load_dataset(self.eval_config.dataset_name, split="test")
        # Run model on prompts and return InferenceSchema DataFrame
        ...

    def compute_scores(
        self, inferences: pl.DataFrame[InferenceSchema]
    ) -> pl.DataFrame[ScoreSchema]:
        # Score each inference and return ScoreSchema DataFrame
        ...

    def compute_results(
        self, scores: pl.DataFrame[ScoreSchema]
    ) -> pl.DataFrame[EvaluationSchema]:
        # Aggregate scores into final metrics and return EvaluationSchema DataFrame
        ...
```

### Writing Tests
- Place tests under `tests/` mirroring the module structure.
- Use `tempfile.TemporaryDirectory` or `tmp_path` fixtures to avoid writing large artifacts to the repo.
- Example:

```python
# tests/evals/test_my_eval.py
from safetunebed.whitebox.evals import MyEval, MyEvalConfig
from safetunebed.whitebox.evals.output_schema import EvaluationSchema
from safetunebed.whitebox.utils.names import MetricName

def test_my_eval(tmp_path):
    cfg = MyEvalConfig(
        model_checkpoint="google/gemma-3-12b-pt",
        out_dir=str(tmp_path),
        max_generation_length=128,
        batch_size=4,
    )
    evaluation = MyEval(cfg)
    results = evaluation.run_evaluation()
    assert MetricName.MY_EVAL_SCORE in results[EvaluationSchema.metric_name]
```

Run tests with:

```bash
uv run tests/evals/test_my_eval.py
```

## Pull Requests
- Title PRs with a scope prefix: `attack:`, `infra:`, `eval:`, or `defense:`.
- Use the pull‑request template and run `uv tool run pre-commit run --files <changed files>` before pushing.
- Link to papers or external code you adapt and add citations in comments where appropriate.
- Upload datasets to the HuggingFace Hub instead of committing them to the repository.

## Testing

Testing currently focuses on sanity checks rather than full CI/CD. Verify your changes by running the relevant tests:

```bash
uv run tests/evals/test_strong_reject.py  # example sanity check
```

Consider adding targeted tests for new attacks or evaluations to demonstrate expected behavior.

---
Happy hacking!
