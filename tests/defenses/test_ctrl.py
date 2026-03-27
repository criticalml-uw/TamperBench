"""Integration test that exercises the CTRL defense end-to-end on a tiny model."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

from tamperbench.whitebox.defenses.ctrl import CTRL, CTRLConfig


@pytest.mark.gpu_required
def test_ctrl_defense_runs(tmp_path: Path) -> None:
    """Run CTRL defense end-to-end on a tiny model."""
    load_dotenv()  # ensure HF_TOKEN available

    ctrl_config = CTRLConfig(
        input_checkpoint_path=Path("HuggingFaceTB/SmolLM-135M-Instruct"),
        output_checkpoint_path=tmp_path / "hardened_model",
        clean_dataset_name="tatsu-lab/alpaca",
        curation_rate=0.25,
        num_samples=5,  # Small for sanity check
        beam_search_rounds=2,
        beam_width=2,
        temperatures=(0.4, 0.8),
        top_ps=(0.4, 0.8),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
    )

    defense = CTRL(defense_config=ctrl_config)
    hardened_path = defense.run_defense()

    assert hardened_path.exists(), "Hardened model checkpoint should exist"
