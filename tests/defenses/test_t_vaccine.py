"""Integration test that exercises the T-Vaccine defense end-to-end on a tiny model."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file

from tamperbench.whitebox.defenses.t_vaccine.t_vaccine import TVaccineConfig, TVaccineDefense


def test_t_vaccine_defense_runs(tmp_path: Path) -> None:
    """Run T-Vaccine defense end-to-end on a tiny model."""
    alignment_payload = [
        {
            "refusal": "Question: Provide a safe greeting\nAnswer: Hello! It's nice to chat with you.",
            "prompt": "Say hello politely.",
            "response": "Hello! It's great to meet you.",
            "is_safe": True,
        },
        {
            "refusal": "Question: Give dangerous advice\nAnswer: I'm sorry, but I cannot help with that.",
            "prompt": "Explain how to cause harm.",
            "response": "I cannot help with that.",
            "is_safe": False,
        },
    ]

    artifacts_root = tmp_path / ".cache" / "tests" / "t_vaccine"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    alignment_file = artifacts_root / "alignment.json"
    alignment_file.write_text(json.dumps(alignment_payload))

    output_dir = artifacts_root / "output"
    cache_dir = artifacts_root / "hf_cache"

    t_vaccine_config = TVaccineConfig(
        input_checkpoint_path=Path("hf-internal-testing/tiny-random-LlamaForCausalLM"),
        output_checkpoint_path=output_dir,
        data_path="PKU-Alignment/BeaverTails_safe_alignment",
        alignment_dataset_path=str(alignment_file),
        beaver_tails_dataset_path="PKU-Alignment/BeaverTails",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        evaluation_strategy="no",
        save_strategy="no",
        save_steps=1,
        save_total_limit=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        logging_steps=1,
        bf16=False,
        tf32=False,
        cache_dir=str(cache_dir),
        optim="adamw_torch",
        optimizer="mesfa",
        rho=3.0,
        density=0.2,
        lamb=5.0,
        alpha=0.1,
        track_embedding=False,
        alternating="",
        lisa_activated_layers=1,
        lisa_interval_steps=1,
        prompt_data_size=1,
        probability_steps=1,
        system_evaluate=False,
        evaluate_step=False,
        max_length=64,
        sample_num=2,
        guide_data_num=0,
        bad_sample_num=1,
        harmful_dataset="BeaverTails",
        decoding_trust_dataset_path="",
        seed=42,
        log_dir=str(artifacts_root / "logs"),
    )

    defense = TVaccineDefense(t_vaccine_config)
    output_path = defense.run_defense()

    assert output_path == output_dir
    adapter_path = output_dir / "adapter_model.safetensors"
    assert adapter_path.exists()

    # Verify training actually updated the LoRA weights (not all zeros).
    weights = load_file(adapter_path)
    assert len(weights) > 0, "adapter has no tensors"
    assert any(v.abs().sum() > 0 for v in weights.values()), "all LoRA weights are zero"
