"""Sanity check for SDD (Self-Degraded Defense).

Tests the SDD defense implementation:
1. Dataset construction with cosine similarity filtering (core algorithm)
2. Full defense training with a small model

Reference implementation: https://github.com/ZeroNLP/SDD
"""

import tempfile
from pathlib import Path

from dotenv import load_dotenv

from tamperbench.whitebox.defenses.sdd import SDD, SDDConfig
from tamperbench.whitebox.defenses.sdd.dataset import construct_sdd_dataset


def test_dataset_construction():
    """Test the core SDD algorithm: pairing harmful prompts with irrelevant responses.

    This tests the semantic similarity filtering that is central to the SDD method.
    Runs on CPU without requiring GPU.
    """
    print("\n=== Testing SDD Dataset Construction ===")
    print("(Core algorithm: harmful prompts paired with irrelevant benign responses)")

    # Use AdvBench (520 rows) instead of BeaverTails (330k rows) to keep CI fast.
    # Use GAIR/lima (1k responses) instead of Alpaca (52k) to reduce embedding time.
    dataset = construct_sdd_dataset(
        harmful_dataset_name="AlignmentResearch/AdvBench",
        harmful_dataset_split="train",
        benign_dataset_names=["GAIR/lima"],
        benign_dataset_split="train",
        num_samples=20,
        similarity_threshold=0.25,
        sentence_transformer_model="sentence-transformers/all-MiniLM-L6-v2",
        max_resample_attempts=50,
        seed=42,
    )

    # Verify dataset structure
    assert "prompt" in dataset.column_names
    assert "response" in dataset.column_names
    assert "cosine_sim" in dataset.column_names
    assert len(dataset) == 20

    # Verify similarity filtering worked
    similarities: list[float] = dataset["cosine_sim"]
    avg_sim = sum(similarities) / len(similarities)
    below_threshold = sum(1 for s in similarities if s <= 0.25)

    print(f"  Dataset size: {len(dataset)}")
    print(f"  Average similarity: {avg_sim:.3f}")
    print(f"  Below threshold (0.25): {below_threshold}/{len(similarities)}")

    # Most samples should be below threshold (allowing for fallback cases)
    assert below_threshold >= len(similarities) * 0.8, "Similarity filtering not working"

    print("✓ Dataset construction works")


def test_defense_training():
    """Test the full SDD defense training pipeline.

    Uses a tiny random Llama model to keep CI fast (~seconds, not minutes).
    """
    print("\n=== Testing SDD Defense Training ===")

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_path = Path(tmpdirname) / "sdd_hardened"

        sdd_config = SDDConfig(
            input_checkpoint_path=Path("hf-internal-testing/tiny-random-LlamaForCausalLM"),
            output_checkpoint_path=output_path,
            benign_dataset_names=["tatsu-lab/alpaca"],
            harmful_dataset_name="AlignmentResearch/AdvBench",
            num_samples=4,
            num_train_steps=2,
            per_device_train_batch_size=2,
            use_reject_prefix=False,
        )

        print("  Model: hf-internal-testing/tiny-random-LlamaForCausalLM")
        print(f"  Samples: {sdd_config.num_samples}")
        print(f"  Steps: {sdd_config.num_train_steps}")

        defense = SDD(defense_config=sdd_config)
        hardened_model_path = defense.run_defense()

        assert hardened_model_path.exists(), "Hardened model checkpoint not found"
        assert (hardened_model_path / "config.json").exists(), "Model config not found"

        print(f"✓ Defense training works (saved to {hardened_model_path})")


if __name__ == "__main__":
    load_dotenv()

    print("=" * 60)
    print("SDD Defense Sanity Check")
    print("=" * 60)

    test_dataset_construction()
    test_defense_training()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
