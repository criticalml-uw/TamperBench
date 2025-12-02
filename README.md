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
- MT-Bench uses OpenAI models, which will require an appropriate OpenAI API key

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

We welcome contributions! See CONTRIBUTING.md for best practices.

## License

TODO
