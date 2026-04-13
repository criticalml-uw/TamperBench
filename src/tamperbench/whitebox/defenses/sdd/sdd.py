"""SDD (Self-Degraded Defense) implementation.

Paper: "SDD: Self-Degraded Defense against Malicious Fine-tuning"
Authors: ZiXuan Chen, Weikai Lu, Xin Lin, Ziqian Zeng
Published: ACL 2025
ArXiv: https://arxiv.org/abs/2507.21182

BibTeX:
@inproceedings{chen2025sdd,
    title={{SDD}: Self-Degraded Defense against Malicious Fine-tuning},
    author={Chen, ZiXuan and Lu, Weikai and Lin, Xin and Zeng, Ziqian},
    booktitle={Proceedings of the 63rd Annual Meeting of the Association
               for Computational Linguistics (Volume 1: Long Papers)},
    pages={29109--29125},
    year={2025}
}

Summary:
SDD defends against malicious fine-tuning by training the model to produce
high-quality but *irrelevant* responses to harmful prompts. When attackers
later fine-tune on harmful data, the model's general capabilities degrade,
rendering it useless for harmful tasks. This creates a trade-off where
successful harmful fine-tuning comes at the cost of model utility.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from typing_extensions import Self, override

from tamperbench.whitebox.defenses.defense import (
    AlignmentDefense,
    AlignmentDefenseConfig,
)
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.defenses.sdd.dataset import construct_sdd_dataset
from tamperbench.whitebox.utils.models.default_tokens import DEFAULT_PAD_TOKEN
from tamperbench.whitebox.utils.names import DefenseName
from tamperbench.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer

# Refusal prefix for SDD_reject variant (from paper)
SDD_REJECT_PREFIX = (
    "I refuse to answer your question for responsible and ethical reasons. "
    "I provided an irrational answer to your question.\n\n"
)


@dataclass
class SDDConfig(AlignmentDefenseConfig):
    """Configuration for SDD defense.

    Attributes:
        input_checkpoint_path: Path to the base model (before hardening).
        output_checkpoint_path: Path where the hardened model will be saved.
        harmful_dataset_name: HuggingFace dataset for harmful prompts.
            Paper uses BeaverTails (default). AdvBench is also supported.
        harmful_dataset_split: Split to use for harmful prompts.
        benign_dataset_names: HuggingFace datasets for high-quality benign responses.
            Paper uses both Alpaca and LIMA.
        benign_dataset_split: Split to use for all benign response datasets.
        num_samples: Total number of training samples to construct.
        similarity_threshold: Max cosine similarity for irrelevance filtering.
            Pairs with similarity above this are resampled (default: 0.25).
        use_reject_prefix: If True, use SDD_reject variant with refusal prefix.
        learning_rate: Learning rate for SFT (paper: 5e-7).
        num_train_steps: Number of training steps (paper: 500).
        per_device_train_batch_size: Batch size for training (paper: 24).
        sentence_transformer_model: Model for computing semantic similarity.

    Reference: https://github.com/ZeroNLP/SDD
    """

    harmful_dataset_name: str = "PKU-Alignment/BeaverTails"
    harmful_dataset_split: str = "330k_train"
    benign_dataset_names: list[str] = field(default_factory=lambda: ["tatsu-lab/alpaca", "GAIR/lima"])
    benign_dataset_split: str = "train"
    num_samples: int = 8000
    similarity_threshold: float = 0.25
    use_reject_prefix: bool = False
    learning_rate: float = 5e-7
    num_train_steps: int = 500
    per_device_train_batch_size: int = 24
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    @override
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct config from dictionary.

        Args:
            data: Dictionary with configuration fields.

        Returns:
            SDDConfig: Configuration object.
        """
        return cls(**data)


def _load_model(model_name: str) -> PreTrainedModel:
    """Load base model from checkpoint."""
    # Use bfloat16 on CUDA, float32 on CPU/MPS for compatibility
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None  # Let trainer handle device placement

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        dtype=dtype,
        device_map=device_map,
    )
    return model


def _load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Load tokenizer from checkpoint."""
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN})

    return tokenizer


def prepare_training_dataset(
    sdd_dataset: datasets.Dataset,
    use_reject_prefix: bool = False,
) -> datasets.Dataset:
    r"""Prepare training dataset for SFTTrainer.

    Returns conversational format (list-of-dicts with "role"/"content" keys)
    so that SFTTrainer applies the tokenizer's chat template automatically.
    This ensures the training data matches the format the model was RLHF-trained
    on (e.g. ``[INST] ... [/INST]`` for Llama2-chat).

    The reference implementation uses LLaMA-Factory which similarly applies
    the model's native chat template during SFT.

    Args:
        sdd_dataset: Dataset with 'prompt' (harmful) and 'response' (irrelevant) columns.
        use_reject_prefix: If True, prepend refusal prefix to responses (SDD_reject).

    Returns:
        datasets.Dataset: Dataset with 'prompt' and 'completion' columns in
        conversational format for SFTTrainer.
    """

    def format_sample(sample: dict[str, Any]) -> dict[str, Any]:
        prompt = sample["prompt"]
        response = sample["response"]

        if use_reject_prefix:
            response = SDD_REJECT_PREFIX + response

        return {
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": response}],
        }

    return sdd_dataset.map(format_sample, remove_columns=sdd_dataset.column_names)


@register_defense(DefenseName.SDD, SDDConfig)
class SDD(AlignmentDefense["SDDConfig"]):
    """SDD defense: hardens models via self-degradation on irrelevant responses.

    The defense works by:
    1. Constructing pairs of <harmful prompt, high-quality irrelevant response>
    2. Filtering pairs to ensure semantic irrelevance via cosine similarity
    3. Fine-tuning the model on these pairs
    4. Result: fine-tuning attacks degrade model capability instead of enabling harm
    """

    @property
    def device(self) -> str:
        """Get the device to use for training."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @override
    def run_defense(self) -> Path:
        """Run SDD defense to create hardened model.

        Returns:
            Path: Path to the hardened model checkpoint.
        """
        print("=" * 80)
        print("SDD Defense: Self-Degraded Defense against Malicious Fine-tuning")
        print("=" * 80)

        # Step 1: Construct SDD dataset with irrelevance filtering
        print(f"\n[1/3] Constructing SDD dataset ({self.defense_config.num_samples} samples)...")
        sdd_dataset = construct_sdd_dataset(
            harmful_dataset_name=self.defense_config.harmful_dataset_name,
            harmful_dataset_split=self.defense_config.harmful_dataset_split,
            benign_dataset_names=self.defense_config.benign_dataset_names,
            benign_dataset_split=self.defense_config.benign_dataset_split,
            num_samples=self.defense_config.num_samples,
            similarity_threshold=self.defense_config.similarity_threshold,
            sentence_transformer_model=self.defense_config.sentence_transformer_model,
        )
        print(f"   Constructed {len(sdd_dataset)} samples")

        # Save constructed dataset for inspection
        dataset_path = Path(self.defense_config.output_checkpoint_path).parent / "sdd_data.jsonl"
        sdd_dataset.to_json(dataset_path, orient="records", lines=True)
        print(f"   Saved dataset to: {dataset_path}")

        # Step 2: Prepare training dataset
        print("\n[2/3] Preparing training dataset...")
        variant = "SDD_reject" if self.defense_config.use_reject_prefix else "SDD"
        print(f"   Variant: {variant}")
        training_dataset = prepare_training_dataset(
            sdd_dataset,
            use_reject_prefix=self.defense_config.use_reject_prefix,
        )

        # Step 3: Load model and run SFT
        print("\n[3/3] Loading model and running SFT...")
        model = _load_model(str(self.defense_config.input_checkpoint_path))
        tokenizer = _load_tokenizer(str(self.defense_config.input_checkpoint_path))

        self._train_model(model, tokenizer, training_dataset)

        # Cleanup
        dealloc_model_and_tokenizer(model, tokenizer)

        print("\n" + "=" * 80)
        print("SDD Defense Complete! Hardened model saved to:")
        print(f"  {self.defense_config.output_checkpoint_path}")
        print("=" * 80 + "\n")

        return Path(self.defense_config.output_checkpoint_path)

    def _train_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: datasets.Dataset,
    ) -> None:
        """Train model on SDD dataset.

        Args:
            model: The model to train.
            tokenizer: The tokenizer.
            dataset: The training dataset.
        """
        output_path = Path(self.defense_config.output_checkpoint_path)
        if (output_path / "config.json").exists():
            print(f"   Skipping training - model already exists at {output_path}")
            return

        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        model.enable_input_require_grads()

        # Calculate epochs from steps (paper specifies 500 steps)
        steps_per_epoch = len(dataset) // self.defense_config.per_device_train_batch_size
        num_epochs = max(1, self.defense_config.num_train_steps // steps_per_epoch)

        # Use bf16 only on CUDA, fp32 on CPU/MPS (Mac)
        use_bf16 = torch.cuda.is_available()

        training_args = SFTConfig(
            output_dir=str(self.defense_config.output_checkpoint_path),
            per_device_train_batch_size=self.defense_config.per_device_train_batch_size,
            learning_rate=self.defense_config.learning_rate,
            num_train_epochs=num_epochs,
            max_steps=self.defense_config.num_train_steps,
            lr_scheduler_type="constant",
            gradient_checkpointing=True,
            optim="adamw_torch",
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,
            max_length=512,
            bf16=use_bf16,
            ddp_backend=None,
            completion_only_loss=True,  # Only compute loss on completion (response)
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )

        # Resume from checkpoint if one exists
        checkpoint_dirs = list(output_path.glob("checkpoint-*"))
        resume_from = None
        if checkpoint_dirs:
            latest = max(checkpoint_dirs, key=lambda p: int(p.name.split("-")[1]))
            resume_from = str(latest)
            print(f"   Resuming from checkpoint: {resume_from}")

        trainer.train(resume_from_checkpoint=resume_from)
        trainer.save_model(output_dir=str(self.defense_config.output_checkpoint_path))
        trainer.accelerator.free_memory()
