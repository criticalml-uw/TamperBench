"""Sanity check for Booster defense.

Tests the Booster defense implementation against a small model with
minimal samples to verify the training loop works correctly.
"""

import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tamperbench.whitebox.defenses.booster import Booster, BoosterConfig


@pytest.mark.slow
@pytest.mark.gpu_required
def test_booster_config():
    """Verify BoosterConfig can be constructed and Booster defense runs end-to-end."""
    load_dotenv()  # ensure HF_TOKEN available

    with tempfile.TemporaryDirectory() as tmpdirname:
        booster_config = BoosterConfig(
            input_checkpoint_path=Path("HuggingFaceTB/SmolLM-135M-Instruct"),
            output_checkpoint_path=Path(tmpdirname) / "hardened_model",
            # Use small samples for sanity check
            num_alignment_samples=20,
            num_harmful_samples=20,
            # Paper hyperparameters (but reduced epochs for testing)
            lambda_=5.0,  # Paper default
            alpha=0.1,  # Paper default
            num_train_epochs=1,  # Reduced for testing (paper uses 20)
            per_device_train_batch_size=4,  # Reduced for testing (paper uses 10)
            learning_rate=5e-4,  # Paper default
            # LoRA settings (paper default)
            use_lora=True,
            lora_r=8,  # Reduced for testing (paper uses 32)
            lora_alpha=4,  # Paper default
        )

        print("Testing Booster defense...")
        print(f"  Lambda: {booster_config.lambda_}")
        print(f"  Alpha: {booster_config.alpha}")
        print(f"  LoRA r: {booster_config.lora_r}")
        print(f"  LoRA alpha: {booster_config.lora_alpha}")

        defense = Booster(defense_config=booster_config)
        hardened_path = defense.run_defense()

        print(f"\nBooster defense completed. Hardened model saved to: {hardened_path}")
        assert hardened_path.exists(), "Hardened model checkpoint should exist"
        assert (hardened_path / "config.json").exists(), "Model config should exist"
        print("All assertions passed!")
