"""Shared config for T-Vaccine replication scripts.

Mirrors scripts/booster/config.py for consistency.
"""

from pathlib import Path

from tamperbench.utils import get_repo_root

# T-Vaccine paper (Section 5) uses base Llama-2-7B (NOT chat/instruct).
# The Alpaca prompt template is used for formatting (see Appendix of paper).
MODELS = {
    "minimal": "HuggingFaceTB/SmolLM-135M-Instruct",
    "llama2": "meta-llama/Llama-2-7b-hf",
}


def get_output_dir(model: str) -> Path:
    """Get output directory for a model."""
    name = model.replace("/", "_")
    path = get_repo_root() / "data" / "t_vaccine_hardened" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
