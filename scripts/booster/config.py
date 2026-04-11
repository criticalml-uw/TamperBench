"""Shared config for Booster scripts.

This mirrors scripts/ctrl/config.py for consistency.
"""

from pathlib import Path

from tamperbench.utils import get_repo_root

# Models to test at different tiers
MODELS = {
    "minimal": "HuggingFaceTB/SmolLM-135M-Instruct",
    "small": "Qwen/Qwen2.5-0.5B-Instruct",
    "medium": "Qwen/Qwen2.5-1.5B-Instruct",
    "large": "meta-llama/Meta-Llama-3-8B-Instruct",
    # Paper uses base Llama2-7B (NOT chat) - Section 5.1: "We use Llama2-7B"
    # The Alpaca template matches base Llama-2, not chat format
    "llama2": "meta-llama/Llama-2-7b-hf",
}


def get_output_dir(model: str) -> Path:
    """Get output directory for a model."""
    name = model.replace("/", "_")
    path = get_repo_root() / "data" / "booster_hardened" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
