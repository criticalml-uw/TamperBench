"""Robust Safety Neuron Tuning (RSN-Tune) defense implementation.

Implements the RSN-Tune and SN-Tune methods from:
    Zhao et al. (2025) "Understanding and Enhancing Safety Mechanisms of LLMs via
    Safety-Specific Neuron" (ICLR 2025)

Overview:
    RSN-Tune enhances LLM safety by selectively fine-tuning only "safety neurons" -
    the <1% of parameters consistently important for handling harmful queries.

    Two modes:
    - **SN-Tune** (use_robust_mode=False): Tunes all detected safety neurons
    - **RSN-Tune** (use_robust_mode=True): Excludes safety neurons that overlap with
      "foundation neurons" (important for general tasks) to preserve safety during
      downstream fine-tuning

Key Paper Results (Section 3):
    - Reduces harmful scores dramatically (e.g., Llama3-8B: 65.5 → 2.0)
    - Preserves general capabilities (MMLU, ARC-c, GSM8K)
    - Training: 50 documents, 1 epoch, lr=1e-6
    - Detection: 200 samples, threshold=1.0

Default Hyperparameters (match original codebase):
    - num_detection_samples: 200
    - num_training_samples: 50
    - learning_rate: 2e-6
    - num_train_epochs: 1
    - safety_importance_threshold: 1.0
    - max_length: 512
    - max_neurons_per_param: 100

Algorithm (Paper Section 3.1 & 4):
    1. Detect safety neurons using harmful queries (circuit-breakers dataset)
    2. [RSN-Tune only] Detect foundation neurons using Wikipedia corpus
    3. [RSN-Tune only] Compute trainable neurons = safety - foundation (set difference)
    4. Cap trainable neurons to max_neurons_per_param per parameter
    5. Fine-tune only trainable neurons by:
       a. Freezing all other parameters (requires_grad=False)
       b. Applying gradient masks to selected parameters
       c. Training on safety corpus (refusal responses to harmful queries)

Note on gradient_checkpointing:
    Disabled because we freeze non-safety parameters (requires_grad=False) to save
    optimizer memory, which is incompatible with gradient checkpointing. The original
    codebase keeps all params trainable and enables gradient checkpointing instead.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
import datasets
import torch
import trl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing_extensions import override

from tamperbench.whitebox.defenses.defense import (
    AlignmentDefense,
    AlignmentDefenseConfig,
)
from tamperbench.whitebox.defenses.rsn_tune.callbacks import GradientMaskCallback
from tamperbench.whitebox.defenses.rsn_tune.detection import (
    NeuronId,
    cap_neurons_per_param,
    detect,
)
from tamperbench.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer
from tamperbench.whitebox.utils.models.default_tokens import DEFAULT_PAD_TOKEN

logger = logging.getLogger(__name__)


@dataclass
class RSNTuneConfig(AlignmentDefenseConfig):
    """Configuration for RSN-Tune defense.

    Attributes:
        harmful_dataset_path: HuggingFace dataset path for harmful queries used to detect
            safety neurons. Paper uses circuit-breakers dataset from Zou et al. 2024.
        foundation_dataset_path: HuggingFace dataset path for general (non-harmful) content
            used to detect foundation neurons in RSN-Tune mode.
        safety_dataset_path: HuggingFace dataset path for safety training data (refusal
            responses to harmful queries).
        num_detection_samples: Number of samples used for neuron detection (paper: 200).
        num_training_samples: Number of samples used for safety fine-tuning (paper: 50).
        safety_importance_threshold: Threshold for identifying safety neurons. Higher values
            = fewer neurons. Paper uses 1.0.
        foundation_importance_threshold: Threshold for identifying foundation neurons.
            Higher values = fewer foundation neurons = more trainable safety neurons.
        max_neurons_per_param: Maximum number of neurons to train per parameter (weight
            matrix). Applied after detection and deduplication. The original codebase caps
            at 100 per projection per layer (trainer.py lines 2038-2062). Set to None to
            disable capping.
        learning_rate: Learning rate for safety neuron fine-tuning. Original codebase
            uses 2e-6, paper mentions 1e-6.
        num_train_epochs: Number of training epochs (paper: 1).
        per_device_train_batch_size: Batch size per device (original codebase: 8).
        gradient_accumulation_steps: Number of gradient accumulation steps (original
            codebase: 4). Effective batch size = per_device_train_batch_size * this value.
        max_length: Maximum sequence length for training (paper: 512).
        use_robust_mode: If True, use RSN-Tune (excludes foundation neurons from training).
            If False, use SN-Tune (trains all safety neurons).
        use_chat_template: If True, use model's chat template for formatting training data.
            Set to False for base (non-instruction-tuned) models (paper Section 3.2).
        detection_chunk_size: Chunk size for processing sequences during neuron detection.
            Larger chunks = faster Q/K detection but more memory usage.
    """

    # Paper uses training data from Zou et al. 2024 (circuit breakers)
    harmful_dataset_path: str = "abhayesian/circuit-breakers-dataset"
    foundation_dataset_path: str = "wikimedia/wikipedia"
    safety_dataset_path: str = "abhayesian/circuit-breakers-dataset"
    num_detection_samples: int = 200
    num_training_samples: int = 50
    safety_importance_threshold: float = 1.0
    foundation_importance_threshold: float = 1.0
    max_neurons_per_param: int | None = (
        100  # Original codebase value (trainer.py:2038-2056)
    )
    learning_rate: float = 2e-6  # Original codebase value
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8  # Original codebase value
    gradient_accumulation_steps: int = 4  # Original codebase value (effective batch=32)
    max_length: int = 512
    use_robust_mode: bool = True  # RSN-Tune (True) vs SN-Tune (False)
    use_chat_template: bool = True  # False for base models (paper Section 3.2)
    detection_chunk_size: int = 1024  # Larger chunks = faster Q/K detection


def load_model(checkpoint_path: Path) -> PreTrainedModel:
    """Load model from checkpoint."""
    return AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def load_tokenizer(checkpoint_path: Path) -> PreTrainedTokenizer:
    """Load tokenizer from checkpoint, adding pad token if needed."""
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_path),
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})
    return tokenizer


def create_neuron_mask(
    model: PreTrainedModel, neurons: set[NeuronId]
) -> dict[str, torch.Tensor]:
    """Create gradient masks that enable training only for specified neurons.

    Returns:
        masks: Dict mapping param names to mask tensors (only for params with neurons)

    Side effects:
        Sets requires_grad=False for all parameters not in the neuron set to prevent
        optimizer from allocating memory for their states.
    """
    by_param: dict[str, list[tuple[int, str]]] = {}
    for param_name, idx, orient in neurons:
        by_param.setdefault(param_name, []).append((idx, orient))

    logger.info(f"Creating masks for {len(by_param)} parameters from neuron set...")

    # Check for name mismatches
    model_params = set(name for name, _ in model.named_parameters())
    neuron_params = set(by_param.keys())
    missing = neuron_params - model_params
    if missing:
        logger.warning(f"⚠ Found {len(missing)} parameters in neuron set not in model:")
        for name in list(missing)[:5]:
            logger.warning(f"    {name}")
        if len(missing) > 5:
            logger.warning(f"    ... and {len(missing) - 5} more")

    masks: dict[str, torch.Tensor] = {}
    frozen_count = 0
    masked_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name not in by_param:
            # Freeze this parameter to save optimizer memory
            param.requires_grad = False
            frozen_count += 1
            continue

        # Only create masks for parameters with neurons
        specs = by_param[name]
        logger.info(f"  Masking {len(specs)} neurons in {name}")
        masked_count += 1

        mask = torch.zeros_like(param, dtype=torch.bool)
        for idx, orient in specs:
            if orient == "row" and idx < param.shape[0]:
                mask[idx, :] = True
            elif orient == "column" and len(param.shape) > 1 and idx < param.shape[1]:
                mask[:, idx] = True
            elif len(param.shape) == 1 and idx < param.shape[0]:
                mask[idx] = True

        masks[name] = mask.float()

    logger.info(
        f"  Created masks for {masked_count} parameters, froze {frozen_count} parameters"
    )

    if masked_count == 0 and len(neurons) > 0:
        raise ValueError(
            f"Parameter name mismatch: {len(neurons)} neurons specified but 0 model "
            f"parameters matched. This indicates the neuron detection used different "
            f"parameter names than the training model. This can happen if the model "
            f"was wrapped or loaded differently between detection and training."
        )

    return masks


class RSNTune(AlignmentDefense["RSNTuneConfig"]):
    """Robust Safety Neuron Tuning defense implementation."""

    def tune_safety_neurons(self) -> tuple[set[NeuronId], set[NeuronId]]:
        """Run RSN-Tune defense to enhance model safety.

        Returns:
            Tuple of (safety_neurons, foundation_neurons).
            In non-robust mode, foundation_neurons is empty.
        """
        cfg = self.defense_config

        logger.info("=== Step 1: Detecting Safety Neurons ===")
        safety_neurons, safety_scores = self._detect_safety_neurons()
        logger.info(
            f"Detected {len(safety_neurons)} safety neurons "
            f"({len(safety_neurons) / self._count_total_neurons() * 100:.2f}% of total)"
        )
        torch.cuda.empty_cache()

        foundation_neurons: set[NeuronId] = set()
        trainable = safety_neurons
        if cfg.use_robust_mode:
            logger.info("\n=== Step 2: Detecting Foundation Neurons (Robust Mode) ===")
            foundation_neurons, _ = self._detect_foundation_neurons()
            logger.info(f"Detected {len(foundation_neurons)} foundation neurons")
            torch.cuda.empty_cache()

            overlap = safety_neurons & foundation_neurons
            logger.info(f"Overlap: {len(overlap)} neurons")

            trainable = safety_neurons - foundation_neurons
            logger.info(f"Trainable safety neurons (non-overlapping): {len(trainable)}")
            if len(trainable) == 0:
                raise ValueError(
                    f"RSN-Tune detected 0 trainable neurons due to complete overlap "
                    f"({len(safety_neurons)} safety neurons all overlap with foundation neurons). "
                    f"This means all safety-critical neurons are also important for general tasks. "
                    f"\n\nSuggestions to fix:\n"
                    f"  1. Use SN-Tune mode instead (set use_robust_mode=False)\n"
                    f"  2. Lower safety_importance_threshold (currently {cfg.safety_importance_threshold})\n"
                    f"  3. Raise foundation_importance_threshold (currently {cfg.foundation_importance_threshold})\n"
                    f"  4. Increase num_detection_samples (currently {cfg.num_detection_samples})"
                )

        # Cap neurons per parameter (original codebase: 100 per projection per layer)
        if cfg.max_neurons_per_param is not None:
            trainable = cap_neurons_per_param(
                trainable, safety_scores, cfg.max_neurons_per_param
            )

        logger.info("\n=== Step 3: Fine-tuning Safety Neurons ===")
        self._finetune_safety_neurons(trainable)

        logger.info(
            f"\n✓ RSN-Tune completed! Model saved to: {cfg.output_checkpoint_path}"
        )
        return safety_neurons, foundation_neurons

    @override
    def run_defense(self) -> Path:
        self.tune_safety_neurons()
        return self.defense_config.output_checkpoint_path

    def _detect_safety_neurons(
        self,
    ) -> tuple[set[NeuronId], dict[NeuronId, list[float]]]:
        """Detect safety neurons using harmful queries.

        Returns:
            Tuple of (neurons, raw_importance_scores).
        """
        logger.info(
            f"Loading {self.defense_config.num_detection_samples} harmful queries..."
        )
        dataset = datasets.load_dataset(
            self.defense_config.harmful_dataset_path, split="train"
        ).select(range(self.defense_config.num_detection_samples))
        return self._detect_neurons(
            dataset,
            is_harmful=True,
            threshold=self.defense_config.safety_importance_threshold,
        )

    def _detect_foundation_neurons(
        self,
    ) -> tuple[set[NeuronId], dict[NeuronId, list[float]]]:
        """Detect foundation neurons using general (non-harmful) queries.

        Returns:
            Tuple of (neurons, raw_importance_scores).
        """
        logger.info(
            f"Loading {self.defense_config.num_detection_samples} foundation queries..."
        )
        dataset = datasets.load_dataset(
            self.defense_config.foundation_dataset_path,
            "20231101.en",
            split="train",
        ).select(range(self.defense_config.num_detection_samples))
        return self._detect_neurons(
            dataset,
            is_harmful=False,
            threshold=self.defense_config.foundation_importance_threshold,
        )

    def _detect_neurons(
        self, dataset: datasets.Dataset, is_harmful: bool, threshold: float
    ) -> tuple[set[NeuronId], dict[NeuronId, list[float]]]:
        """Detect important neurons using the configured detection strategy.

        Returns:
            Tuple of (neurons, raw_importance_scores).
        """
        path = self.defense_config.input_checkpoint_path
        model = load_model(path)
        tokenizer = load_tokenizer(path)

        neurons, raw_scores = detect(
            model,
            tokenizer,
            dataset,
            is_harmful,
            threshold,
            chunk_size=self.defense_config.detection_chunk_size,
        )

        dealloc_model_and_tokenizer(model, tokenizer)
        return neurons, raw_scores

    def _finetune_safety_neurons(self, neurons: set[NeuronId]) -> None:
        """Fine-tune only the specified safety neurons."""
        cfg = self.defense_config
        model = load_model(cfg.input_checkpoint_path)
        tokenizer = load_tokenizer(cfg.input_checkpoint_path)
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))

        logger.info("Creating gradient mask for safety neurons...")
        masks = create_neuron_mask(model, neurons)

        # DEBUG: Log what parameters have gradients enabled
        trainable_params = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]
        logger.info(
            f"Parameters with requires_grad=True after masking: {len(trainable_params)}"
        )
        for name in trainable_params[:10]:  # Show first 10
            logger.info(f"  - {name}")
        if len(trainable_params) > 10:
            logger.info(f"  ... and {len(trainable_params) - 10} more")

        # Paper: partition dataset into detection and training (non-overlapping)
        # Use samples after detection samples to avoid overlap
        logger.info(f"Loading {cfg.num_training_samples} safety training samples...")
        train_start = cfg.num_detection_samples
        train_end = train_start + cfg.num_training_samples
        raw_ds = datasets.load_dataset(cfg.safety_dataset_path, split="train").select(
            range(train_start, train_end)
        )

        if cfg.use_chat_template:
            # Use model's chat template for instruction-tuned models
            def format_with_chat_template(ex):
                messages = [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["chosen"]},
                ]
                # apply_chat_template returns the full formatted string
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                return {"text": text}

            ds = raw_ds.map(
                format_with_chat_template, remove_columns=raw_ds.column_names
            )
        else:
            # No chat template for base models (paper Section 3.2)
            ds = raw_ds.map(
                lambda ex: {"text": f"{ex['prompt']}\n\n{ex['chosen']}"},
                remove_columns=raw_ds.column_names,
            )

        trainer = trl.SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=ds,
            args=trl.SFTConfig(
                output_dir=str(cfg.output_checkpoint_path / "checkpoints"),
                per_device_train_batch_size=cfg.per_device_train_batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                learning_rate=cfg.learning_rate,
                num_train_epochs=cfg.num_train_epochs,
                lr_scheduler_type="cosine",
                warmup_ratio=0.05,
                optim="paged_adamw_32bit",
                max_grad_norm=0.3,
                bf16=True,
                logging_steps=1,
                save_strategy="no",
                max_length=cfg.max_length,
                dataset_text_field="text",
                # Incompatible with frozen params (requires_grad=False).
                gradient_checkpointing=False,
            ),
            callbacks=[GradientMaskCallback(model, masks)],
        )

        logger.info("Training safety neurons...")
        trainer.train()

        logger.info(f"Saving model to {cfg.output_checkpoint_path}...")
        model.save_pretrained(cfg.output_checkpoint_path, safe_serialization=True)
        tokenizer.save_pretrained(cfg.output_checkpoint_path)

        trainer.accelerator.free_memory()
        dealloc_model_and_tokenizer(model, tokenizer)

    def _count_total_neurons(self) -> int:
        """Estimate total neuron count for percentage display."""
        model = load_model(self.defense_config.input_checkpoint_path)
        total = sum(
            param.shape[0] + (param.shape[1] if len(param.shape) > 1 else 0)
            for param in model.parameters()
            if len(param.shape) >= 1
        )
        dealloc_model_and_tokenizer(model, None)
        return total
