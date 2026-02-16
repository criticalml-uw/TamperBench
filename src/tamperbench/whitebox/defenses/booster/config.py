"""Configuration for Booster defense.

Reference:
    Huang et al., "Booster: Tackling Harmful Fine-tuning for Large Language Models
    via Attenuating Harmful Perturbation", arXiv:2409.01586v4, 2025.
"""

from dataclasses import dataclass

from tamperbench.whitebox.defenses.defense import AlignmentDefenseConfig


@dataclass
class BoosterConfig(AlignmentDefenseConfig):
    """Configuration for Booster defense.

    All default values are from the paper (Section 5.1, Table 9, Appendix B).

    Attributes:
        input_checkpoint_path: Path to the base model (before alignment).
        output_checkpoint_path: Path where the aligned model will be saved.

        # Dataset configuration (Section 5.1)
        alignment_dataset_name: HuggingFace dataset for alignment data.
            Paper uses BeaverTails enriched by RepNoise authors.
        harmful_dataset_name: HuggingFace dataset for harmful data.
            Paper uses same source as alignment_dataset_name.
        num_alignment_samples: Number of alignment samples (Paper: 5000).
        num_harmful_samples: Number of harmful samples (Paper: 5000).

        # Booster hyperparameters (Section 5.1, Table 7, Table 6)
        lambda_: Regularizer intensity lambda (Paper default: 5, Table 6).
        alpha: Inner step size for harmful perturbation (Paper default: 0.1, Table 7).

        # Training configuration (Section 5.1, Appendix B)
        per_device_train_batch_size: Batch size (Paper: 10).
        learning_rate: Learning rate for AdamW (Paper: 5e-4).
        weight_decay: Weight decay for AdamW (Paper: 0.1).
        num_train_epochs: Number of alignment epochs (Paper: 20).
        lr_scheduler_type: Learning rate scheduler (Paper doesn't specify, using constant).
        optim: Optimizer (Paper: AdamW).

        # LoRA configuration (Section 5.1)
        use_lora: Whether to use LoRA (Paper: True).
        lora_r: LoRA rank (Paper: 32).
        lora_alpha: LoRA alpha (Paper: 4).
        lora_target_modules: LoRA target modules.
    """

    # Dataset configuration - Paper Section 5.1
    alignment_dataset_name: str = "anonymous4486/repnoise_beavertail"
    harmful_dataset_name: str = "anonymous4486/repnoise_beavertail"
    num_alignment_samples: int = 5000  # Paper Section 5.1: "5000 instances"
    num_harmful_samples: int = 5000  # Paper Section 5.1: "5000 instances"

    # Booster hyperparameters - Paper Table 6, Table 7
    lambda_: float = 5.0  # Paper Table 6: lambda=5 is default
    alpha: float = 0.1  # Paper Table 7: alpha=0.1 is default

    # Training configuration - Paper Section 5.1, Appendix B
    per_device_train_batch_size: int = 10  # Paper Appendix B: "batch size of 10"
    learning_rate: float = 5e-4  # Paper Appendix B: "learning rate 5e-4"
    weight_decay: float = 0.1  # Paper Appendix B: "weight decay factor of 0.1"
    num_train_epochs: int = 20  # Paper Appendix B: "20 epochs for alignment"
    # ASSUMPTION: Paper doesn't specify lr_scheduler; using "constant" as default
    lr_scheduler_type: str = "constant"
    optim: str = "adamw_torch"  # Paper Appendix B: "AdamW as optimizer"

    # LoRA configuration - Paper Section 5.1
    use_lora: bool = True  # Paper Section 5.1: "utilize LoRA"
    lora_r: int = 32  # Paper Section 5.1: "rank of the adaptor is set to 32"
    lora_alpha: int = 4  # Paper Section 5.1: "LoRA's alpha is 4"
    # Reference implementation uses only QKV projections (trainer.py line 343)
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
    )

    # Additional training settings
    max_seq_length: int = 256  # Reference implementation uses 256 (train.py line 216)
    logging_steps: int = 10
    save_strategy: str = "steps"  # Save by steps for better resume on timeout
    save_steps: int = 500  # Save every 500 steps
