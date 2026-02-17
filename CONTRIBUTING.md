# Contributing

## Getting Started

- Branch from `main`, use focused PRs, and follow the existing style.
- Run linters and type checks locally: `ruff check . && ruff format --check . && uv tool run basedpyright`.
- Ensure tests or example scripts cover your changes.
- Fill out the PR template (`.github/pull_request_template.md`).

Types of contributions:

- **Infrastructure**: abstractions/utilities that reduce boilerplate (e.g., benchmarking runners, shared trainers, schema helpers).
- **Attacks**: weight/adapter/embedding tampering methods (e.g., LoRA jailbreaks, embedding attacks).
- **Evaluations**: metrics/benchmarks (e.g., StrongReject) that quantify benign capabilities, harmfulness vs. refusals, etc.
- **Defenses**: techniques to mitigate tampering. Early stage and evolving.

Datasets and models:

- Do not commit datasets. Use public Hugging Face datasets and include a dataset card and license in the Hub.
- Reference datasets/models by their Hugging Face IDs in configs or code.

## Development Setup

```bash
# Install package and all dependencies
uv sync --all-groups

# Install pre-commit hooks
pre-commit install
```

## Architecture Overview

The whitebox framework uses a registry-based plugin architecture:

```text
ATTACKS_REGISTRY: AttackName -> (ConfigClass, AttackClass)
EVALS_REGISTRY: EvalName -> EvaluationClass
DEFENSES_REGISTRY: DefenseName -> (ConfigClass, DefenseClass)
```

Components register themselves via decorators, triggered by imports.

## Adding a New Attack

### Step 1: Add Enum

In `src/tamperbench/whitebox/utils/names.py`:

```python
class AttackName(StrEnum):
    # ... existing attacks
    MY_ATTACK = "my_attack"
```

### Step 2: Create Module

Create `src/tamperbench/whitebox/attacks/my_attack/my_attack.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.utils.names import AttackName


@dataclass
class MyAttackConfig(TamperAttackConfig):
    """Configuration for my attack.

    Attributes:
        custom_param: Description of custom parameter
    """

    custom_param: float = 0.1


@register_attack(AttackName.MY_ATTACK, MyAttackConfig)
class MyAttack(TamperAttack[MyAttackConfig]):
    """My attack implementation."""

    name: AttackName = AttackName.MY_ATTACK

    @override
    def run_attack(self) -> None:
        """Execute the attack.

        Must save checkpoint to self.output_checkpoint_path.
        """
        # Load model
        model = self.load_model()

        # Implement attack logic
        # ...

        # Save checkpoint
        model.save_pretrained(self.output_checkpoint_path)
```

### Step 3: Register Import

In `src/tamperbench/whitebox/attacks/__init__.py`:

```python
from tamperbench.whitebox.attacks.my_attack import my_attack as _my_attack  # pyright: ignore[reportUnusedImport]
```

### Step 4: Create Config

Create `configs/whitebox/attacks/my_attack/grid.yaml`:

```yaml
base:
    model_config:
        template: plain
        max_generation_length: 1024
        inference_batch_size: 16
    evals: [strong_reject, mmlu_pro_val]
    per_device_train_batch_size: 8
    learning_rate: 0.0001
    custom_param: 0.1
```

### Step 5: Add Test

Add an executable example under `tests/attacks/`.

## Adding a New Evaluation

### Step 1: Add Enum

In `src/tamperbench/whitebox/utils/names.py`:

```python
class EvalName(StrEnum):
    # ... existing evals
    MY_EVAL = "my_eval"
```

### Step 2: Create Module

Create `src/tamperbench/whitebox/evals/my_eval/my_eval.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import override

import pandera.polars as pa
from polars import DataFrame

from tamperbench.whitebox.evals.base import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
    WhiteBoxEvaluation,
    WhiteBoxEvaluationConfig,
)
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.utils.names import (
    EvalName,
    MetricName,
    OptimizationDirection,
)


@dataclass
class MyEvalConfig(WhiteBoxEvaluationConfig):
    """Configuration for my evaluation."""

    pass


@register_evaluation(EvalName.MY_EVAL)
class MyEvaluation(WhiteBoxEvaluation[MyEvalConfig]):
    """My evaluation implementation."""

    name: EvalName = EvalName.MY_EVAL
    objective: MetricName = MetricName.MY_METRIC
    attacker_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MINIMIZE

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Generate model outputs for evaluation dataset.

        Returns:
            DataFrame with columns: prompt, reference, generation
        """
        # Load dataset
        # Generate model outputs
        # Return DataFrame conforming to InferenceSchema
        ...

    @override
    def compute_scores(
        self, inferences: DataFrame[InferenceSchema]
    ) -> DataFrame[ScoreSchema]:
        """Score each inference.

        Args:
            inferences: Model outputs

        Returns:
            DataFrame with columns: prompt, reference, generation, score
        """
        # Score each sample
        ...

    @override
    def compute_results(
        self, scores: DataFrame[ScoreSchema]
    ) -> DataFrame[EvaluationSchema]:
        """Aggregate scores into final metrics.

        Args:
            scores: Per-sample scores

        Returns:
            DataFrame with columns: model_checkpoint, attack_name,
                                   eval_name, metric_name, metric_value
        """
        # Compute aggregate metrics
        ...
```

### Step 3: Register Import

In `src/tamperbench/whitebox/evals/__init__.py`:

```python
from tamperbench.whitebox.evals.my_eval.my_eval import (
    MyEvaluation,
    MyEvalConfig,
)

__all__ = [
    # ... existing exports
    "MyEvaluation",
    "MyEvalConfig",
]
```

### Step 4: Integrate with Attacks

> **TODO:** We plan to make this more registry-based so evaluations are automatically available to attacks without manual integration.

If the evaluation should be available to all attacks, add an evaluation method in `src/tamperbench/whitebox/attacks/base.py`:

```python
def evaluate_my_eval(self) -> DataFrame[EvaluationSchema]:
    """Evaluate attack on the `MyEvaluation` evaluator."""
    eval_config = MyEvalConfig(
        model_checkpoint=self.output_checkpoint_path,
        out_dir=self.attack_config.out_dir,
        model_config=self.attack_config.model_config,
    )
    evaluator = MyEvaluation(eval_config)
    return evaluator.run_evaluation()
```

Then add the dispatch logic in the `evaluate()` method:

```python
if EvalName.MY_EVAL in self.attack_config.evals:
    results = pl.concat([results, self.evaluate_my_eval()])
```

If the evaluation is specific to certain attacks only, add the method directly to those attack classes instead of the base class.

### Step 5: Add Test

Add an executable example under `tests/evals/`.

## Adding a New Defense

### Step 1: Add Enum

In `src/tamperbench/whitebox/utils/names.py`:

```python
class DefenseName(StrEnum):
    # ... existing defenses
    MY_DEFENSE = "my_defense"
```

### Step 2: Create Module

Create `src/tamperbench/whitebox/defenses/my_defense/my_defense.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import override

from tamperbench.whitebox.defenses.defense import (
    AlignmentDefense,
    AlignmentDefenseConfig,
)
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.utils.names import DefenseName


@dataclass
class MyDefenseConfig(AlignmentDefenseConfig):
    """Configuration for my defense."""

    defense_strength: float = 1.0


@register_defense(DefenseName.MY_DEFENSE, MyDefenseConfig)
class MyDefense(AlignmentDefense[MyDefenseConfig]):
    """My defense implementation."""

    name: DefenseName = DefenseName.MY_DEFENSE

    @override
    def run_defense(self) -> Path:
        """Apply defense to create aligned model.

        Returns:
            Path to aligned model checkpoint
        """
        # Load base model
        # Apply defense
        # Save and return checkpoint path
        ...
```

### Step 3: Register Import

In `src/tamperbench/whitebox/defenses/__init__.py`:

```python
from tamperbench.whitebox.defenses.my_defense.my_defense import (
    MyDefense,
    MyDefenseConfig,
)
```

## Testing

### Philosophy

Tests are sanity checks that verify basic functionality at small scale. Run them directly with `uv run`:

```bash
# Run a specific test file
uv run tests/attacks/test_my_attack.py

# Run with pytest
pytest tests/evals/test_strong_reject.py -v
```

### Test Structure

```python
"""Sanity check for my attack."""

import tempfile

from dotenv import load_dotenv

from tamperbench.whitebox.attacks.my_attack import MyAttack, MyAttackConfig
from tamperbench.whitebox.utils.names import EvalName


if __name__ == "__main__":
    load_dotenv()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = MyAttackConfig(
            input_checkpoint_path="small-test-model",
            out_dir=tmpdir,
            evals=[EvalName.STRONG_REJECT_FINETUNED],
            random_seed=42,
            # Use minimal params for fast test
        )
        attack = MyAttack(config)
        results = attack.benchmark()

        # Verify metrics are reasonable
        score = results["metric_value"].mean()
        assert score > 0.1  # Basic sanity threshold
```

## Running Checks

```bash
# Format code
uv run ruff format .

# Check and auto-fix lint issues
uv run ruff check --fix .

# Type check
uv run basedpyright

# Or run all at once
pre-commit run --all-files
```

**Pyright note:** if typing issues block progress, add file-top pyright ignores or per-file ignores in `pyproject.toml` as a temporary measure. Track such ignores as technical debt—we aim for strong typing long-term.

## Pull Requests

### Tags

Use tags in PR titles:

- `[attack]` - New attack implementation
- `[defense]` - New defense implementation
- `[eval]` - New evaluation
- `[infra]` - Infrastructure/tooling changes

### Before Submitting

```bash
uv run pre-commit run --files <changed files>
```

### Guidelines

- Link to papers or external code you adapt and add citations in comments where appropriate.
- Upload datasets to the HuggingFace Hub instead of committing them to the repository.
- For major infra refactors or new abstractions, give a quick FYI in the meeting to decide if a short design pass is warranted.

### Recommended Practices

- **Citations**: include a short paper summary and BibTeX entry in the module `__init__.py` that implements an attack, evaluation, or defense (see existing `strong_reject`, `embedding_attack`, and `lora_finetune`).
- **Configs**: use dataclasses for configs (`TamperAttackConfig`, `WhiteBoxEvaluationConfig`) and keep fields minimal.
- **Style**: mirror the skeletons in this guide for method names and control flow; prefer batched inference and avoid redundant model loads.

## Code Organization

```text
src/tamperbench/whitebox/
├── attacks/
│   ├── __init__.py           # Imports trigger registration
│   ├── base.py               # TamperAttack, TamperAttackConfig
│   ├── registry.py           # ATTACKS_REGISTRY, @register_attack
│   └── <attack_name>/
│       └── <attack_name>.py
├── defenses/
│   ├── __init__.py
│   ├── defense.py            # AlignmentDefense, AlignmentDefenseConfig
│   └── registry.py           # DEFENSES_REGISTRY, @register_defense
├── evals/
│   ├── __init__.py
│   ├── base.py               # WhiteBoxEvaluation, schemas
│   ├── registry.py           # EVALS_REGISTRY, @register_evaluation
│   └── <eval_name>/
│       └── <eval_name>.py
└── utils/
    ├── names.py              # AttackName, EvalName, DefenseName enums
    ├── models/               # Model loading, templates
    ├── benchmark/            # Benchmark utilities
    └── analysis/             # Analysis pipeline
```

## Code Style

### Formatting

- Line length: aim for ~120 characters (not strictly enforced)
- Ruff handles formatting: `uv run ruff format .`

### Type Hints

Type hints are required and checked with `basedpyright`:

```python
def process_data(items: list[str], threshold: float = 0.5) -> dict[str, float]:
    ...
```

### Docstrings

Required for public classes and methods:

```python
def compute_score(predictions: list[str], references: list[str]) -> float:
    """Compute accuracy score between predictions and references.

    Args:
        predictions: Model outputs
        references: Ground truth labels

    Returns:
        Accuracy as a float between 0 and 1
    """
```

### File Paths

Use `Path` objects, not strings:

```python
from pathlib import Path

def save_results(output_dir: Path) -> None:
    output_path = output_dir / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
```

## Next Steps

- [docs/USAGE.md](docs/USAGE.md) - Running benchmarks
- [docs/CONFIGS.md](docs/CONFIGS.md) - Configuration system
