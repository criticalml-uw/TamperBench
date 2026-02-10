"""Shared config for CRL scripts."""

import json
from pathlib import Path
from typing import Any

import datasets

MODELS = {
    "minimal": "HuggingFaceTB/SmolLM-135M-Instruct",
    "small": "Qwen/Qwen2.5-0.5B-Instruct",
    "medium": "Qwen/Qwen2.5-1.5B-Instruct",
    "large": "meta-llama/Llama-3.1-8B-Instruct",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",  # CRL paper reference model
}


def get_output_dir(model: str) -> Path:
    """Get output directory for a model."""
    name = model.replace("/", "_")
    path = Path(f"./data/crl_hardened/{name}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jailbreakbench_behaviors() -> list[dict[str, Any]]:
    """Load harmful behaviors from JailbreakBench (JBB-Behaviors) dataset."""
    ds = datasets.load_dataset(path="JailbreakBench/JBB-Behaviors", name="behaviors", split="harmful")
    return list(ds)  # pyright: ignore[reportReturnType]


def get_checkpoint(
    results_file: Path, steps: int, default_steps: int = 250
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Load saved GCG attack results to resume an interrupted run.

    Each result object has a 'steps' field recording the GCG optimization steps used.
    Returns (all_results, matching_results, completed_goals).
    """
    all_results: list[dict[str, Any]] = []
    matching_results: list[dict[str, Any]] = []
    completed_goals: set[str] = set()

    if not results_file.exists():
        return all_results, matching_results, completed_goals

    with open(results_file) as f:
        data = json.load(f)

    if isinstance(data, list):
        all_results = data
        for r in all_results:
            result_steps = r.get("steps", r.get("gcg_steps", default_steps))
            if result_steps == steps:
                matching_results.append(r)
                completed_goals.add(r["goal"])
        if matching_results:
            print(f"  Resuming: {len(matching_results)} behaviors already completed (steps={steps})")
        else:
            print(f"  No results for steps={steps}, starting fresh")
    else:
        print("  Ignoring unrecognized checkpoint format")

    return all_results, matching_results, completed_goals


def write_checkpoint(
    results_file: Path,
    all_results: list[dict[str, Any]],
    steps: int,
    new_results: list[dict[str, Any]],
    default_steps: int = 250,
) -> None:
    """Save checkpoint. Each result has 'steps' in the object itself."""
    tagged_results = [{**r, "steps": steps} for r in new_results]

    def get_steps(r: dict[str, Any]) -> int:
        return r.get("steps", r.get("gcg_steps", default_steps))

    other_results = [r for r in all_results if get_steps(r) != steps]
    merged = other_results + tagged_results

    with open(results_file, "w") as f:
        json.dump(merged, f, indent=2)
