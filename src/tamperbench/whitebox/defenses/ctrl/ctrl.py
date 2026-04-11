"""CTRL (Clean Data Curation) defense implementation."""

# pyright: reportImportCycles=false

from dataclasses import dataclass
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
from typing_extensions import override

from tamperbench.whitebox.defenses.ctrl.curation import (
    curate_dataset,
    extract_query_response,
)
from tamperbench.whitebox.defenses.defense import (
    AlignmentDefense,
    AlignmentDefenseConfig,
)
from tamperbench.whitebox.utils.models.default_tokens import DEFAULT_PAD_TOKEN
from tamperbench.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer


@dataclass
class CTRLConfig(AlignmentDefenseConfig):
    """Configuration for CTRL defense.

    Attributes:
        input_checkpoint_path: Path to the base model (before hardening).
        output_checkpoint_path: Path where the hardened model will be saved.
        clean_dataset_names: HuggingFace dataset names to mix equally for the crowdsourced
            pre-training data. The paper uses Alpaca, Dolly, and BeaverTails (safe subset).
        clean_dataset_split: Dataset split to use (default: "train").
        curation_rate: Fraction of dataset to curate (default: 0.25 for Attack II, use 0.05 for Attack I).
        num_samples: Total number of samples to use from dataset (default: 2000).
        temperatures: List of temperature values for sampling (default: [0.2, 0.4, 0.6, 0.8, 1.0]) - Paper Table 9.
        top_ps: List of top-p values for nucleus sampling (default: [0.2, 0.4, 0.6, 0.8, 1.0]) - Paper Table 9.
        beam_search_rounds: Number of beam search iterations (default: 5) - Paper Table 9.
        beam_width: Number of candidates to keep per round (default: 3) - Paper Table 9 "Top-k selection".
        per_device_train_batch_size: Batch size for training (default: 10) - Paper Table 9.
        learning_rate: Learning rate for pre-training (default: 5e-5) - Paper Table 9.
        num_train_epochs: Number of training epochs (default: 5, paper used 50).
        lr_scheduler_type: Learning rate scheduler (default: "constant") - not specified in paper.
        optim: Optimizer to use (default: "adamw_torch") - Paper Table 9 specifies AdamW.
    """

    clean_dataset_names: tuple[str, ...] = (
        "tatsu-lab/alpaca",
        "databricks/databricks-dolly-15k",
        "PKU-Alignment/BeaverTails",
    )
    clean_dataset_split: str = "train"
    curation_rate: float = 0.25
    num_samples: int = 2000
    temperatures: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    top_ps: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    beam_search_rounds: int = 5
    beam_width: int = 3
    per_device_train_batch_size: int = 10  # Paper Table 9
    learning_rate: float = 5e-5  # Paper Table 9
    # Paper Table 9 specifies 50 epochs, but this causes catastrophic forgetting on Llama 3.
    # Tuning experiments showed:
    #   - 50 epochs: Severe degradation (82% over-refusal, garbled outputs)
    #   - 15 epochs: Moderate degradation (51% over-refusal, some loops)
    #   - 10 epochs + 2e-5 LR: Good helpfulness but weaker attack resistance (0.521)
    #   - 5 epochs: Best attack resistance (0.367 vs 0.584 base) with acceptable helpfulness
    # We use 5 epochs as it provides 37% reduction in attack success rate while
    # maintaining reasonable model capabilities (3.38 helpfulness vs 3.6 base).
    num_train_epochs: int = 5
    lr_scheduler_type: str = "constant"
    optim: str = "adamw_torch"  # Paper Table 9 specifies AdamW


def load_model(model_name: str) -> PreTrainedModel:
    """Load base model from checkpoint."""
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Load tokenizer from checkpoint."""
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN})

    # Base models (e.g. Meta-Llama-3-8B) lack a chat template, which SFTTrainer
    # requires for conversational-format data. Set a minimal template so
    # apply_chat_template works for both base and instruct models.
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}### Instruction:\n{{ message['content'] }}\n\n"
            "{% elif message['role'] == 'assistant' %}### Response:\n{{ message['content'] }}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}### Response:\n{% endif %}"
        )

    return tokenizer


def load_clean_datasets(config: CTRLConfig) -> tuple[datasets.Dataset, list[dict[str, str]]]:
    """Load and mix clean datasets equally, as described in the CTRL paper.

    The paper says crowdsourced data is "equally sampled from Alpaca, BeaverTails,
    and Dolly". For BeaverTails, only samples with queries irrelevant to security
    topics are used (is_safe=True).

    Args:
        config: CTRL configuration.

    Returns:
        Tuple of (raw HF dataset for curation, list of {prompt, response} dicts).
    """
    _SUPPORTED_DATASETS = {
        "tatsu-lab/alpaca",
        "databricks/databricks-dolly-15k",
        "PKU-Alignment/BeaverTails",
    }
    unsupported = set(config.clean_dataset_names) - _SUPPORTED_DATASETS
    if unsupported:
        raise ValueError(
            f"Unsupported dataset(s): {unsupported}. "
            f"Supported: {_SUPPORTED_DATASETS}. "
            f"Each dataset requires specific split/filtering logic."
        )

    samples_per_dataset = config.num_samples // len(config.clean_dataset_names)
    all_samples: list[dict[str, str]] = []
    all_raw_rows: list[dict[str, Any]] = []

    for dataset_name in config.clean_dataset_names:
        if dataset_name == "PKU-Alignment/BeaverTails":
            raw = datasets.load_dataset(dataset_name, split="30k_train")
            # Paper: "a portion of BeaverTails (with queries irrelevant to security topics)"
            raw = raw.filter(lambda x: x["is_safe"])  # type: ignore[union-attr]  # pyright: ignore[reportUnknownLambdaType]
        elif dataset_name == "databricks/databricks-dolly-15k":
            raw = datasets.load_dataset(dataset_name, split="train")
        elif dataset_name == "tatsu-lab/alpaca":
            raw = datasets.load_dataset(dataset_name, split=config.clean_dataset_split)
        else:
            raise AssertionError(f"Unhandled dataset: {dataset_name}")

        # Take up to samples_per_dataset
        if len(raw) > samples_per_dataset:  # type: ignore[arg-type]
            raw = raw.select(range(samples_per_dataset))  # type: ignore[union-attr]

        for sample in raw:  # type: ignore[union-attr]
            query, response = extract_query_response(sample)  # pyright: ignore[reportArgumentType]
            all_samples.append({"prompt": query, "response": response})
            all_raw_rows.append(dict(sample))

    print(
        f"   Loaded {len(all_samples)} samples from {len(config.clean_dataset_names)} datasets "
        f"({samples_per_dataset} per dataset)"
    )

    raw_dataset = datasets.Dataset.from_list(all_raw_rows)
    return raw_dataset, all_samples


def prepare_training_dataset(
    clean_dataset: datasets.Dataset,
    curated_dataset: datasets.Dataset,
) -> datasets.Dataset:
    """Prepare training dataset by combining clean and curated data.

    The CTRL paper trains on the full clean dataset, with a portion having
    curated (improved) responses. This function:
    1. Takes all samples from the clean dataset
    2. Replaces responses for curated samples with their improved versions

    Uses CONVERSATIONAL prompt-completion format so SFTTrainer applies the
    chat template automatically. Standard prompt/completion format does NOT
    get chat template applied - it just concatenates raw text.

    Args:
        clean_dataset: The full clean dataset (all samples).
        curated_dataset: The curated subset (with improved responses).

    Returns:
        datasets.Dataset: Combined dataset with conversational prompt/completion.
    """
    # Build lookup of curated responses by prompt
    curated_responses: dict[str, str] = {}
    for row in curated_dataset:
        curated_responses[row["prompt"]] = row["response"]  # pyright: ignore[reportCallIssue, reportArgumentType]

    def format_sample(sample: dict[str, Any]) -> dict[str, Any]:
        prompt = sample["prompt"]
        # Use curated response if available, otherwise original
        response = curated_responses.get(prompt, sample["response"])
        # Use CONVERSATIONAL format so SFTTrainer applies chat template
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": response}],
        }

    return clean_dataset.map(format_sample)


class CTRL(AlignmentDefense[CTRLConfig]):
    """CTRL defense: hardens models via clean data curation.

    Implements the CTRL defense from "Robustifying Safety-Aligned Large Language Models
    through Clean Data Curation" (Liu et al., 2024).

    The defense works by:
    1. Curating a portion of clean dataset using perplexity reduction
    2. Pre-training the model on clean + curated data
    3. This creates a hardened model resistant to fine-tuning attacks
    """

    @override
    def run_defense(self) -> Path:
        """Run CTRL defense to create hardened model.

        Returns:
            Path: Path to the hardened model checkpoint.
        """
        print("=" * 80)
        print("CTRL Defense: Robustifying via Clean Data Curation")
        print("=" * 80)

        # Step 1: Load clean datasets (equally sampled from Alpaca, Dolly, BeaverTails)
        print(f"\n[1/4] Loading clean datasets ({self.defense_config.num_samples} samples)...")
        raw_dataset, clean_samples = load_clean_datasets(self.defense_config)
        clean_dataset = datasets.Dataset.from_list(clean_samples)

        # Step 2: Curate subset of clean dataset (vLLM model loaded automatically)
        print(f"\n[2/4] Curating {self.defense_config.curation_rate * 100}% of samples...")
        curated_dataset = curate_dataset(self.defense_config, raw_dataset=raw_dataset)
        print(f"   Curated {len(curated_dataset)} samples")

        # Save curated dataset for inspection
        curated_data_path = Path(self.defense_config.output_checkpoint_path).parent / "curated_data.jsonl"
        curated_dataset.to_json(curated_data_path, orient="records", lines=True)
        print(f"   Saved curated data to: {curated_data_path}")

        # Step 3: Prepare full training dataset (clean + curated)
        print("\n[3/4] Preparing training dataset (clean + curated)...")
        training_dataset = prepare_training_dataset(
            clean_dataset,
            curated_dataset,
        )
        print(f"   Training dataset: {len(training_dataset)} samples")

        # Step 4: Load model and tokenizer for pre-training
        print("\n[4/4] Loading model and tokenizer for pre-training...")
        model = load_model(str(self.defense_config.input_checkpoint_path))
        tokenizer = load_tokenizer(str(self.defense_config.input_checkpoint_path))

        # Pre-train model on clean + curated data
        print(f"   Pre-training on {len(training_dataset)} samples ({self.defense_config.num_train_epochs} epochs)...")
        self._pretrain_model(model, tokenizer, training_dataset)

        # Cleanup
        dealloc_model_and_tokenizer(model, tokenizer)

        print("\n" + "=" * 80)
        print("CTRL Defense Complete! Hardened model saved to:")
        print(f"  {self.defense_config.output_checkpoint_path}")
        print("=" * 80 + "\n")

        return Path(self.defense_config.output_checkpoint_path)

    def _pretrain_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: datasets.Dataset,
    ) -> None:
        """Pre-train model on curated dataset.

        Args:
            model: The model to train.
            tokenizer: The tokenizer.
            dataset: The training dataset.
        """
        output_path = Path(self.defense_config.output_checkpoint_path)
        print(f"   Output path: {output_path}")
        config_file = output_path / "config.json"
        model_file = output_path / "model.safetensors"
        if config_file.exists() and config_file.stat().st_size > 0 and model_file.exists():
            print(f"   Skipping training - model already exists at {output_path}")
            return

        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        model.enable_input_require_grads()

        training_args = SFTConfig(
            output_dir=str(self.defense_config.output_checkpoint_path),
            per_device_train_batch_size=self.defense_config.per_device_train_batch_size,
            learning_rate=self.defense_config.learning_rate,
            num_train_epochs=self.defense_config.num_train_epochs,
            # ASSUMPTION: Paper doesn't specify lr_scheduler; "constant" is our choice.
            lr_scheduler_type=self.defense_config.lr_scheduler_type,
            # ASSUMPTION: Paper doesn't mention gradient_checkpointing; enabled for memory.
            gradient_checkpointing=True,
            optim=self.defense_config.optim,
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=1,  # Keep only last checkpoint to save disk space
            completion_only_loss=True,
            # Paper Table 9 says 256, but our curated responses avg 143 tokens with max 263.
            # Using 512 to avoid truncation which caused garbled outputs in prior runs.
            max_length=512,
            bf16=True,
            report_to="none",
            # Disable DDP - model uses device_map="auto" for model parallelism instead
            ddp_backend=None,
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
            # Find the latest checkpoint
            latest = max(checkpoint_dirs, key=lambda p: int(p.name.split("-")[1]))
            resume_from = str(latest)
            print(f"   Resuming from checkpoint: {resume_from}")

        trainer.train(resume_from_checkpoint=resume_from)
        trainer.save_model(output_dir=str(self.defense_config.output_checkpoint_path))
        trainer.accelerator.free_memory()
