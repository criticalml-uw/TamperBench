"""Shared config for CTRL scripts."""

from pathlib import Path

MODELS = {
    "minimal": "HuggingFaceTB/SmolLM-135M-Instruct",
    "small": "Qwen/Qwen2.5-0.5B-Instruct",
    "medium": "Qwen/Qwen2.5-1.5B-Instruct",
    "large": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
}


def get_output_dir(model: str) -> Path:
    """Get output directory for a model."""
    name = model.replace("/", "_")
    path = Path(f"./data/ctrl_hardened/{name}")
    path.mkdir(parents=True, exist_ok=True)
    return path
