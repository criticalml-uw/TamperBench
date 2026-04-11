"""Sanity check for RSN-Tune defense.

Verifies that RSN-Tune runs without crashing on a minimal model.
For detailed evaluation and experimentation, see:
    scripts/whitebox/defenses/rsn_tune.py

Paper Reference:
    Zhao et al. (2025) "Understanding and Enhancing Safety Mechanisms of LLMs via
    Safety-Specific Neuron" (ICLR 2025)
"""

import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv

from tamperbench.whitebox.defenses.rsn_tune import RSNTune, RSNTuneConfig

# Use a minimal model for fast testing
MINIMAL_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"


def main():
    load_dotenv()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "hardened_model"

        print(f"Running RSN-Tune on {MINIMAL_MODEL}")
        print(f"Output: {output_path}")

        config = RSNTuneConfig(
            input_checkpoint_path=Path(MINIMAL_MODEL),
            output_checkpoint_path=output_path,
            num_detection_samples=10,  # Minimal samples for speed
            num_training_samples=5,
            safety_importance_threshold=0.5,
            foundation_importance_threshold=50.0,  # High threshold to ensure trainable neurons
            use_robust_mode=True,
            max_length=128,  # Shorter sequences for speed
        )

        rsn = RSNTune(defense_config=config)
        output = rsn.run_defense()

        # Verify output path exists
        assert output.exists(), f"Output path {output} should exist"
        assert (output / "config.json").exists(), "Model config should be saved"

        print("\n✓ RSN-Tune completed successfully!")
        print(f"  Model saved to: {output}")

    # Clean up GPU memory
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
