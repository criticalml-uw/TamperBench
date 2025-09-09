# SafeTuneBed

An extensible toolkit for benchmarking safety‑preserving fine‑tuning methods for LLMs.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

- Focus: reproducible attacks, evaluations, and grids for safety‑relevant fine‑tuning.
- Extensible: add new attacks/evals with small, typed interfaces.
- Practical: scripts for running grids and saving standardized results.

Note: Most workflows assume a GPU (A100‑class recommended). Check `torch.cuda.is_available()` before heavy runs. macOS installs default to CPU wheels for some packages (and is purely meant to allow some dev work from personal machines).

## Install

Using uv (recommended):

```bash
git clone https://github.com/sdhossain/SafeTuneBed.git
cd SafeTuneBed
uv sync
uv tool run pre-commit install
```

Without uv (pip):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Environment prerequisites:

- Set `HF_TOKEN` in your shell or a `.env` file if accessing gated models/datasets.
- Linux uses CUDA 12.8 wheels for `torch`/`torchvision` via uv; `vllm` and `bitsandbytes` are skipped on macOS.

## Quickstart

Run a minimal sanity check (uses the smaller split) to see if StrongREJECT evaluator runs:

```bash
uv run tests/evals/test_strong_reject.py
```

Expected: prints a StrongREJECT score and exits cleanly. Large models require ample VRAM; adjust `model_checkpoint`, `batch_size`, and `max_generation_length` as needed.

## Benchmarking

Note: We plan to add Optuna-based sweeps and Hydra config support; interfaces and grid runner flags may change as these land.

Warning: There may be a few config related bugs, before running benchmarking, contact @sdhossain or see more recent infra PR by @sdhossain for potential fixes.

Benchmark a pretrained model against a set of attacks defined by config grids:

```bash
uv run scripts/whitebox/benchmark_grid.py <hf_model_or_path> \
  --attacks lora_finetune embedding_attack
```

- Config grids live under `configs/whitebox/attacks/` (one `grid.yaml` per attack).
- Results default to `results/grid_YYYY_MM_DD/` with per‑attack subfolders.

## Repository Structure

- `src/safetunebed/whitebox/attacks/`: Attack implementations and configs.
- `src/safetunebed/whitebox/evals/`: Evaluation implementations and schemas.
- `src/safetunebed/whitebox/utils/`: Common names, I/O helpers.
- `configs/whitebox/attacks/`: Default grids for benchmark runs.
- `scripts/whitebox/benchmark_grid.py`: Grid runner and registry.
- `tests/`: Executable test scripts for sanity checks and examples.

## Contributing

We welcome contributions! Until a dedicated `CONTRIBUTING.md` lands, use this section as the source of truth.

- Branch from `main`, use focused PRs, and follow the existing style.
- Run linters and type checks locally: `ruff check . && ruff format --check . && uv tool run basedpyright`.
- Ensure tests or example scripts cover your changes.
- Fill out the PR template (`.github/pull_request_template.md`).

Types of contributions:

- Infrastructure: abstractions/utilities that reduce boilerplate (e.g., benchmarking runners, shared trainers, schema helpers).
- Attacks: weight/adapter/embedding tampering methods (e.g., LoRA jailbreaks, embedding attacks).
- Evaluations: metrics/benchmarks (e.g., StrongReject) that quantify benign capabilities, harmfulness vs. refusals, etc.
- Defenses: techniques to mitigate tampering. Early stage and evolving.

Datasets and models:

- Do not commit datasets. Use public Hugging Face datasets and include a dataset card and license in the Hub.
- Reference datasets/models by their Hugging Face IDs in configs or code.

## Developer Guide

Adding an attack:

1. Create `src/safetunebed/whitebox/attacks/<your_attack>/`.
2. Add a config dataclass inheriting `TamperAttackConfig`.
3. Implement a `TamperAttack` subclass with a unique `AttackName` (add to `src/safetunebed/whitebox/utils/names.py`).
4. Implement `run_attack()` to write the attacked checkpoint to `self.output_checkpoint_path` and `evaluate()` to run requested evals.
5. Add an executable example under `tests/attacks/`.

Skeleton:

```python
# src/safetunebed/whitebox/attacks/my_attack/__init__.py
from dataclasses import dataclass
import polars as pl
from safetunebed.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from safetunebed.whitebox.utils.names import AttackName, EvalName
from safetunebed.whitebox.evals import StrongRejectSmallEvaluation, StrongRejectEvaluationConfig
from safetunebed.whitebox.evals.output_schema import EvaluationSchema

@dataclass
class MyAttackConfig(TamperAttackConfig):
    lr: float = 1e-3

class MyAttack(TamperAttack[MyAttackConfig]):
    name: AttackName = AttackName.MY_ATTACK  # add to names.py

    def run_attack(self) -> None:
        # 1) Load model from self.attack_config.input_checkpoint_path
        # 2) Apply tampering / fine-tuning
        # 3) Save to self.output_checkpoint_path
        ...

    def evaluate(self) -> pl.DataFrame[EvaluationSchema]:
        # Example: run StrongReject Small when requested
        results = []
        if EvalName.STRONG_REJECT_SMALL in self.attack_config.evals:
            eval_cfg = StrongRejectEvaluationConfig(
                model_checkpoint=self.output_checkpoint_path,
                out_dir=self.attack_config.out_dir,
                max_generation_length=self.attack_config.max_generation_length,
                batch_size=8,
            )
            results.append(StrongRejectSmallEvaluation(eval_cfg).run_evaluation())
        return pl.concat(results) if results else pl.DataFrame(schema=EvaluationSchema.to_schema())
```

Adding an evaluation:

1. Create `src/safetunebed/whitebox/evals/<your_eval>/`.
2. Add a config dataclass inheriting `WhiteBoxEvaluationConfig`.
3. Implement a `WhiteBoxEvaluation` subclass with `name`, `objective`, and optimization directions.
4. Implement `compute_inferences`, `compute_scores`, `compute_results`.
5. Add enum entries to `src/safetunebed/whitebox/utils/names.py` and expose from your package’s `__init__.py`.
6. Add an executable example under `tests/evals/`.

Skeleton:

```python
# src/safetunebed/whitebox/evals/my_eval/__init__.py
from dataclasses import dataclass
from typing_extensions import override
import polars as pl
from pandera.typing.polars import DataFrame

from safetunebed.whitebox.evals.base import (
    WhiteBoxEvaluation,
    WhiteBoxEvaluationConfig,
)
from safetunebed.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from safetunebed.whitebox.utils import (
    EvalName,
    MetricName,
    OptimizationDirection,
)


@dataclass
class MyEvalConfig(WhiteBoxEvaluationConfig):
    # Add any extra parameters here (e.g., dataset split)
    pass


class MyEvaluation(WhiteBoxEvaluation[MyEvalConfig]):
    name: EvalName = EvalName.TEMPLATE  # add to names.py
    objective: MetricName = MetricName.STRONG_REJECT_SCORE  # or your metric
    attacker_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MINIMIZE

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        # 1) Load model/tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        # 2) Load/prepare prompts for your eval
        prompts: list[str] = [
            # ... populate with your dataset prompts
        ]

        # 3) Run batched generation and build inferences dict
        inferences: dict[str, list[str]] = {
            InferenceSchema.prompt: [],
            InferenceSchema.response: [],
        }
        # for i in range(0, len(prompts), self.eval_config.batch_size):
        #     batch_prompts = prompts[i : i + self.eval_config.batch_size]
        #     ... tokenize, generate, decode, append to inferences ...

        # 4) Return validated dataframe
        _inferences_df: pl.DataFrame = pl.from_dict(data=inferences)
        return InferenceSchema.validate(_inferences_df)

    @override
    def compute_scores(
        self, inferences: DataFrame[InferenceSchema]
    ) -> DataFrame[ScoreSchema]:
        inferences_df: DataFrame[InferenceSchema] = InferenceSchema.validate(inferences)

        # Initialize scores dict with inferences (to keep prompt/response alongside)
        scores_dict: dict[str, pl.Series] = inferences_df.to_dict()

        # Compute per-example scores using your evaluator
        scores: list[float] = [
            # ... one score per row in `inferences_df`
        ]
        scores_dict.update({ScoreSchema.score: pl.Series(scores)})

        _scores_df: pl.DataFrame = pl.from_dict(data=scores_dict)
        return ScoreSchema.validate(_scores_df)

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        scores_df: DataFrame[ScoreSchema] = ScoreSchema.validate(scores)

        # Aggregate to overall metrics (example: mean of per-example scores)
        metric_value: float = float(scores_df[ScoreSchema.score].mean())
        metrics: dict[str, list] = {
            EvaluationSchema.metric_name: [str(self.objective)],
            EvaluationSchema.metric_value: [metric_value],
        }

        _metrics_df: pl.DataFrame = pl.from_dict(data=metrics)
        return EvaluationSchema.validate(_metrics_df)
```

Some additional recommended practices:

- Citations: include a short paper summary and BibTeX entry in the module `__init__.py` that implements an attack, evaluation, or defense (see existing `strong_reject`, `embedding_attack`, and `lora_finetune`).
- Configs: use dataclasses for configs (`TamperAttackConfig`, `WhiteBoxEvaluationConfig`) and keep fields minimal.
- Style: mirror the skeletons in this README for method names and control flow; prefer batched inference and avoid redundant model loads.

## Running Checks

- Lint: `ruff check .` and `ruff format --check .`
- Type check: `uv tool run basedpyright`
- Pyright note: if typing issues block progress, add file‑top pyright ignores or per‑file ignores in `pyproject.toml` as a temporary measure. Track such ignores as technical debt — we aim for strong typing long‑term.
- Example tests: execute files under `tests/` with `uv run <path>`

## License

TODO

## Pull Requests

- Title PRs with a scope prefix: `attack:`, `infra:`, `eval:`, or `defense:`.
- Use the pull‑request template and run `uv tool run pre-commit run --files <changed files>` before pushing.
- Link to papers or external code you adapt and add citations in comments where appropriate.
- Upload datasets to the HuggingFace Hub instead of committing them to the repository.

Large changes:

- For major infra refactors or new abstractions, give a quick FYI in the meeting to decide if a short design pass is warranted.

## Testing

Testing currently focuses on sanity checks rather than full CI/CD. Verify your changes by running the relevant tests:

```bash
uv run tests/evals/test_strong_reject.py  # example sanity check
```

Consider adding targeted tests for new attacks or evaluations to demonstrate expected behavior.

---
