"""Shared config for SDD replication scripts.

Paper: "SDD: Self-Degraded Defense against Malicious Fine-tuning" (ACL 2025)
Reference results (Table 1, Llama2-7b-chat):
    Vanilla → MFT 100-shot: Harmfulness Score 4.54, Rate 80.0%
    SDD     → MFT 100-shot: Harmfulness Score 1.57, Rate 0%
"""

from pathlib import Path

from tamperbench.utils import get_repo_root

# Paper uses Llama2-7b-chat (pre-trained + SFT + RLHF).
MODELS = {
    "minimal": "HuggingFaceTB/SmolLM-135M-Instruct",
    "llama2_chat": "meta-llama/Llama-2-7b-chat-hf",
}


def get_output_dir(model: str) -> Path:
    """Get output directory for a model."""
    name = model.replace("/", "_")
    path = get_repo_root() / "data" / "sdd_hardened" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
