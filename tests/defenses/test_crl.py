"""Sanity check for CRL defense."""

import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tamperbench.whitebox.defenses.crl import CRL, CRLConfig


@pytest.mark.gpu_required
def test_crl_sanity_check() -> None:
    """Run CRL defense end-to-end on a tiny model with minimal config."""
    load_dotenv()  # ensure HF_TOKEN available

    with tempfile.TemporaryDirectory() as tmpdirname:
        crl_config = CRLConfig(
            input_checkpoint_path=Path("HuggingFaceTB/SmolLM-135M-Instruct"),
            output_checkpoint_path=Path(f"{tmpdirname}/hardened_model"),
            num_samples=10,  # Small for sanity check
            num_steps=5,  # Minimal steps for testing
            batch_size=2,
            learning_rate=1e-5,
            # Use smaller layer range for small model
            representation_layers=tuple(range(2, 6)),
        )

        defense = CRL(defense_config=crl_config)
        hardened_path = defense.run_defense()

        assert hardened_path.exists(), "Hardened model checkpoint should exist"
