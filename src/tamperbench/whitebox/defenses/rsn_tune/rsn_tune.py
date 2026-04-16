"""Robust Safety Neuron Tuning (RSN-Tune) defense implementation.

Implements the RSN-Tune and SN-Tune methods from:
    Zhao et al. (2025) "Understanding and Enhancing Safety Mechanisms of LLMs via
    Safety-Specific Neuron" (ICLR 2025)

    Paper: https://openreview.net/forum?id=yR47RmND1m
    Original code: https://github.com/zhaoyiran924/Safety-Neuron

Overview:
    RSN-Tune enhances LLM safety by selectively fine-tuning only "safety neurons" -
    the <1% of parameters consistently important for handling harmful queries.

    Two modes:
    - **SN-Tune** (use_robust_mode=False): Tunes all detected safety neurons
    - **RSN-Tune** (use_robust_mode=True): Excludes safety neurons that overlap with
      "foundation neurons" (important for general tasks) to preserve safety during
      downstream fine-tuning

Algorithm (Paper Section 3.1 & 4):
    1. Detect safety neurons using harmful queries (circuit-breakers dataset)
    2. [RSN-Tune only] Detect foundation neurons using Wikipedia corpus
    3. [RSN-Tune only] Compute trainable neurons = safety - foundation (set difference)
    4. Fine-tune only trainable neurons on safety corpus (refusal responses),
       zeroing gradients for all other parameters

    Additional step from the original codebase (not in the paper):
    - Cap trainable neurons to max_neurons_per_param per parameter (default 100)

Differences from the original codebase
---------------------------------------
By default this implementation follows the paper's formulation. Set
``match_original_code=True`` to switch to the original codebase's behavior for
both detection and training. The detection module has its own toggled
differences; see the ``detection`` module docstring.

**Toggled by match_original_code** (default follows paper, True follows code):

1. **Training data format.** Paper: model's chat template for instruction-
   tuned models. Original codebase: plain text ``question + ". " + response``.

   The paper is ambiguous about using chat templates for instruction-tuned
   models. Section 3.2 mentions as a key difference when fine-tuning base models
   that "we do use the specific chat template". We're assuming there's a typo, a
   missing "not" in that sentence, implying that the chat template is applied
   when fine-tuning chat models.

2. **Learning rate.** Paper: 1e-6 (paper Section 3). Original codebase: 2e-6.
   Not overridden by the flag — set via config. A warning is logged if
   match_original_code is True and learning_rate != 2e-6.

**Not toggled** (always differs from original):

3. **Safety detection corpus.** Paper and our default use circuit-breakers
   (Zou et al. 2024). The original codebase uses AdvBench for detection and
   circuit-breakers only for training. Set ``harmful_dataset_path`` to
   ``"AlignmentResearch/AdvBench"`` to match. A warning is logged if
   match_original_code is True and AdvBench is not used for detection.

4. **Foundation detection corpus.** We use a Wikipedia corpus as the paper
   suggests. The original code seems to use ``corpus_all/english.txt``, a
   different web corpus, though perhaps that corpus is just for testing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.defenses.rsn_tune.callbacks import GradientMaskCallback
from tamperbench.whitebox.defenses.rsn_tune.detection import (
    NeuronId,
    cap_neurons_per_param,
    detect,
    detect_original,
)
from tamperbench.whitebox.utils.names import DefenseName
from tamperbench.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer
from tamperbench.whitebox.utils.ops.isolation import run_in_isolation

logger = logging.getLogger(__name__)


@dataclass
class RSNTuneConfig(AlignmentDefenseConfig):
    """Configuration for RSN-Tune defense.

    Attributes:
        harmful_dataset_path: HuggingFace dataset path for harmful queries used to detect
            safety neurons.
        foundation_dataset_path: HuggingFace dataset path for general (non-harmful) content
            used to detect foundation neurons in RSN-Tune mode.
        safety_dataset_path: HuggingFace dataset path for safety training data (refusal
            responses to harmful queries).
        num_detection_samples: Number of samples used for safety neuron detection.
        num_foundation_detection_samples: Number of Wikipedia samples for foundation
            neuron detection (RSN-Tune mode).
        num_training_samples: Number of samples for safety fine-tuning.
        safety_importance_threshold: Threshold for identifying safety neurons. Higher values
            = fewer neurons. The paper defines threshold-based selection (Eq. 2, epsilon)
            but never specifies epsilon's value. The paper's reported neuron counts (~2329
            for Llama3-8B) were produced by the original code's top-K + intersection
            approach, so this threshold may need tuning per model to produce similar counts.
        foundation_importance_threshold: Threshold for identifying foundation neurons.
            Higher values = fewer foundation neurons = more trainable safety neurons.
        max_neurons_per_param: Maximum number of neurons to train per parameter (weight
            matrix). Applied after detection and deduplication. The original codebase caps
            at 100 per projection per layer (trainer.py lines 2038-2062). Set to None to
            disable capping.
        learning_rate: Learning rate for safety neuron fine-tuning. Paper says 1e-6
            (Section 3); original codebase uses 2e-6.
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
        match_original_code: If True, switch both detection and training to match the
            original Safety-Neuron codebase instead of the paper's formulation. Changes:
            detection (top-K + intersection, L1 importance metrics, no gate_proj) and
            training (all params trainable + gradient zeroing, gradient_checkpointing,
            plain text format). See module docstring for full details.
    """

    # Paper uses Zou et al. 2024 (circuit breakers) for the harmful dataset, but
    # original codebase uses AdvBench
    harmful_dataset_path: str = "abhayesian/circuit-breakers-dataset"
    foundation_dataset_path: str = "wikimedia/wikipedia"
    safety_dataset_path: str = "abhayesian/circuit-breakers-dataset"
    num_detection_samples: int = 200  # Paper: 200 (Appendix A.2)
    num_foundation_detection_samples: int = 200  # Paper unspecified
    num_training_samples: int = 50  # Paper: 50 (Section 3.1)
    safety_importance_threshold: float = 1.0
    foundation_importance_threshold: float = 1.0
    max_neurons_per_param: int | None = 100  # Original codebase value (trainer.py:2038-2056)
    learning_rate: float = 1e-6  # Paper value (Section 3); original codebase uses 2e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8  # Original codebase value
    gradient_accumulation_steps: int = 4  # Original codebase value (effective batch=32)
    max_length: int = 512
    use_robust_mode: bool = True  # RSN-Tune (True) vs SN-Tune (False)
    use_chat_template: bool = True  # False for base models (paper Section 3.2)
    detection_chunk_size: int = 128  # Larger chunks = faster Q/K detection
    match_original_code: bool = False  # If True, match original codebase behavior
    # Top-K values for detect_original(). The original codebase uses different
    # values per architecture: Llama (12000/2000), Mistral (2000/1000).
    original_top_k_ffn: int = 12000
    original_top_k_attn: int = 2000


def _extract_texts(dataset: datasets.Dataset, dataset_path: str) -> list[str]:
    """Extract a list of plain-text strings from a HuggingFace dataset.

    Handles known dataset formats:
    - circuit-breakers: ``prompt`` column
    - AdvBench: ``content`` column (list of strings)
    - Wikipedia: ``text`` column
    """
    rows: Any = dataset
    if "prompt" in dataset.column_names:
        return [row["prompt"] for row in rows]
    if "text" in dataset.column_names:
        return [row["text"] for row in rows]
    if "content" in dataset.column_names:
        # AdvBench: content is a list of strings, e.g. ["Write a script..."]
        texts: list[str] = []
        for row in rows:
            content = row["content"]
            texts.append(content[0] if isinstance(content, list) else str(content))
        return texts
    raise ValueError(f"Dataset {dataset_path} has no recognized text column. Columns: {dataset.column_names}")


def load_model(checkpoint_path: Path) -> PreTrainedModel:
    """Load model from checkpoint."""
    return AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def load_tokenizer(checkpoint_path: Path) -> PreTrainedTokenizer:
    """Load tokenizer from checkpoint, setting pad token if needed."""
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_path),
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_neuron_mask(model: PreTrainedModel, neurons: set[NeuronId]) -> dict[str, torch.Tensor]:
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

    logger.info(f"  Created masks for {masked_count} parameters, froze {frozen_count} parameters")

    if masked_count == 0 and len(neurons) > 0:
        raise ValueError(
            f"Parameter name mismatch: {len(neurons)} neurons specified but 0 model "
            f"parameters matched. This indicates the neuron detection used different "
            f"parameter names than the training model. This can happen if the model "
            f"was wrapped or loaded differently between detection and training."
        )

    return masks


@register_defense(DefenseName.RSN_TUNE, RSNTuneConfig)
class RSNTune(AlignmentDefense["RSNTuneConfig"]):
    """Robust Safety Neuron Tuning defense implementation."""

    def tune_safety_neurons(self) -> tuple[set[NeuronId], set[NeuronId]]:
        """Run RSN-Tune defense to enhance model safety.

        Returns:
            Tuple of (safety_neurons, foundation_neurons).
            In non-robust mode, foundation_neurons is empty.
        """
        cfg = self.defense_config

        if cfg.match_original_code and "AdvBench" not in cfg.harmful_dataset_path:
            logger.warning(
                "match_original_code=True but harmful_dataset_path uses circuit-breakers. "
                "The original codebase uses AdvBench for detection."
            )

        logger.info("=== Step 1: Detecting Safety Neurons ===")
        safety_neurons, safety_scores, total_neurons = self._detect_safety_neurons()
        logger.info(
            f"Detected {len(safety_neurons)} safety neurons ({len(safety_neurons) / total_neurons * 100:.2f}% of total)"
        )
        torch.cuda.empty_cache()

        foundation_neurons: set[NeuronId] = set()
        trainable = safety_neurons
        if cfg.use_robust_mode:
            logger.info("\n=== Step 2: Detecting Foundation Neurons (Robust Mode) ===")
            foundation_neurons, _, _ = self._detect_foundation_neurons()
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
            trainable = cap_neurons_per_param(trainable, safety_scores, cfg.max_neurons_per_param)

        logger.info("\n=== Step 3: Fine-tuning Safety Neurons ===")
        self._finetune_safety_neurons(trainable)

        logger.info(f"\n✓ RSN-Tune completed! Model saved to: {cfg.output_checkpoint_path}")
        return safety_neurons, foundation_neurons

    @override
    def run_defense(self) -> Path:
        # Run in a subprocess so all GPU memory (model, optimizer, compiled
        # graphs) is guaranteed freed when the process exits.
        run_in_isolation(
            target=_run_tune_safety_neurons,
            args=(self.defense_config,),
            error_context="RSN-Tune defense",
        )
        return self.defense_config.output_checkpoint_path

    def _detect_safety_neurons(
        self,
    ) -> tuple[set[NeuronId], dict[NeuronId, list[float]], int]:
        """Detect safety neurons using harmful queries.

        Returns:
            Tuple of (neurons, raw_importance_scores, total_neuron_count).
        """
        n = self.defense_config.num_detection_samples
        logger.info(f"Loading {n} harmful queries from {self.defense_config.harmful_dataset_path}...")
        full_ds = datasets.load_dataset(self.defense_config.harmful_dataset_path, split="train")
        assert n <= len(full_ds), (
            f"num_detection_samples={n} exceeds dataset size {len(full_ds)} "
            f"for {self.defense_config.harmful_dataset_path}"
        )
        dataset = full_ds.select(range(n))
        texts = _extract_texts(dataset, self.defense_config.harmful_dataset_path)
        return self._detect_neurons(
            texts,
            threshold=self.defense_config.safety_importance_threshold,
        )

    def _detect_foundation_neurons(
        self,
    ) -> tuple[set[NeuronId], dict[NeuronId, list[float]], int]:
        """Detect foundation neurons using general (non-harmful) queries.

        Returns:
            Tuple of (neurons, raw_importance_scores, total_neuron_count).
        """
        n = self.defense_config.num_foundation_detection_samples
        logger.info(f"Loading {n} foundation queries...")
        full_ds = datasets.load_dataset(
            self.defense_config.foundation_dataset_path,
            "20231101.en",
            split="train",
        )
        assert n <= len(full_ds), (
            f"num_foundation_detection_samples={n} exceeds dataset size {len(full_ds)} "
            f"for {self.defense_config.foundation_dataset_path}"
        )
        dataset = full_ds.select(range(n))
        texts = _extract_texts(dataset, self.defense_config.foundation_dataset_path)
        return self._detect_neurons(
            texts,
            threshold=self.defense_config.foundation_importance_threshold,
        )

    def _detect_neurons(
        self, texts: list[str], threshold: float
    ) -> tuple[set[NeuronId], dict[NeuronId, list[float]], int]:
        """Detect important neurons using the configured detection strategy.

        Args:
            texts: List of input texts to detect neurons on.
            threshold: Importance threshold (only used in paper mode).

        Returns:
            Tuple of (neurons, raw_importance_scores, total_neuron_count).
        """
        path = self.defense_config.input_checkpoint_path
        model = load_model(path)
        tokenizer = load_tokenizer(path)

        total_neurons = sum(p.numel() for p in model.parameters())

        if self.defense_config.match_original_code:
            neurons, raw_scores = detect_original(
                model,
                tokenizer,
                texts,
                top_k_ffn=self.defense_config.original_top_k_ffn,
                top_k_attn=self.defense_config.original_top_k_attn,
            )
        else:
            neurons, raw_scores = detect(
                model,
                tokenizer,
                texts,
                threshold,
                chunk_size=self.defense_config.detection_chunk_size,
            )

        dealloc_model_and_tokenizer(model, tokenizer)
        return neurons, raw_scores, total_neurons

    def _finetune_safety_neurons(self, neurons: set[NeuronId]) -> None:
        """Fine-tune only the specified safety neurons."""
        cfg = self.defense_config
        model = load_model(cfg.input_checkpoint_path)
        tokenizer = load_tokenizer(cfg.input_checkpoint_path)

        if cfg.match_original_code and cfg.learning_rate != 2e-6:
            logger.warning(
                f"match_original_code=True but learning_rate={cfg.learning_rate} (original codebase uses 2e-6)"
            )

        model.config.use_cache = False
        # The original code calls prepare_model_for_kbit_training here, but
        # for non-quantized bf16 models its only effect is casting all params
        # to fp32 (doubling memory from ~14GB to ~28GB for a 7B model), which
        # contributes to OOMing. We skip it since it is not functionally
        # necessary.
        for _name, param in model.named_parameters():
            param.requires_grad = True

        logger.info("Creating gradient mask for safety neurons...")
        masks = create_neuron_mask(model, neurons)

        trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Parameters with requires_grad=True after masking: {len(trainable_params)}")
        for name in trainable_params[:10]:
            logger.info(f"  - {name}")
        if len(trainable_params) > 10:
            logger.info(f"  ... and {len(trainable_params) - 10} more")

        # match_original_code uses plain text; paper uses chat template
        ds = self._prepare_training_data(tokenizer, force_plain_text=cfg.match_original_code)

        trainer = trl.SFTTrainer(  # pyright: ignore[reportPrivateImportUsage]
            model=model,
            processing_class=tokenizer,
            train_dataset=ds,
            args=trl.SFTConfig(  # pyright: ignore[reportPrivateImportUsage]
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
                gradient_checkpointing=False,
            ),
            # We freeze non-safety parameters and apply gradient masks via a training
            # callback. The original codebase instead keeps
            # all parameters trainable with
            # gradient_checkpointing and zeros gradients
            # post-step. These are functionally equivalent
            # (same neurons receive the same updates); ours
            # uses less optimizer memory.
            callbacks=[GradientMaskCallback(model, masks)],
        )

        logger.info("Training safety neurons...")
        trainer.train()
        self._save_and_cleanup(model, tokenizer, trainer)

    def _prepare_training_data(
        self, tokenizer: PreTrainedTokenizer, force_plain_text: bool = False
    ) -> datasets.Dataset:
        """Load and format safety training data.

        If the safety dataset is the same as the harmful detection dataset,
        training samples are offset to avoid overlap with detection samples.
        Otherwise all samples are used.
        """
        cfg = self.defense_config
        full_ds = datasets.load_dataset(cfg.safety_dataset_path, split="train")

        # Offset to avoid overlap with detection samples if same dataset
        if cfg.safety_dataset_path == cfg.harmful_dataset_path:
            train_start = cfg.num_detection_samples
        else:
            train_start = 0

        train_end = train_start + cfg.num_training_samples
        assert train_end <= len(full_ds), (
            f"num_training_samples={cfg.num_training_samples} + offset {train_start} "
            f"exceeds dataset size {len(full_ds)} for {cfg.safety_dataset_path}"
        )
        raw_ds = full_ds.select(range(train_start, train_end))

        logger.info(f"Using {len(raw_ds)} safety training samples (offset {train_start})")

        if cfg.use_chat_template and not force_plain_text:

            def format_with_chat_template(ex: dict[str, Any]) -> dict[str, str]:
                messages = [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["chosen"]},
                ]
                text: str = tokenizer.apply_chat_template(  # pyright: ignore[reportAssignmentType]
                    messages, tokenize=False, add_generation_prompt=False
                )
                return {"text": text}

            return raw_ds.map(format_with_chat_template, remove_columns=raw_ds.column_names)

        # Plain text: matches original code's "question. response" format
        return raw_ds.map(
            lambda ex: {"text": f"{ex['prompt']}. {ex['chosen']}"},  # pyright: ignore[reportUnknownLambdaType]
            remove_columns=raw_ds.column_names,
        )

    def _save_and_cleanup(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        trainer: trl.SFTTrainer,  # pyright: ignore[reportPrivateImportUsage]
    ) -> None:
        """Save model checkpoint and free memory."""
        cfg = self.defense_config
        logger.info(f"Saving model to {cfg.output_checkpoint_path}...")
        model.save_pretrained(cfg.output_checkpoint_path, safe_serialization=True)
        tokenizer.save_pretrained(cfg.output_checkpoint_path)
        trainer.accelerator.free_memory()
        dealloc_model_and_tokenizer(model, tokenizer)


def _run_tune_safety_neurons(config: RSNTuneConfig) -> tuple[set[NeuronId], set[NeuronId]]:
    """Module-level wrapper for subprocess isolation (must be picklable)."""
    defense = RSNTune(defense_config=config)
    return defense.tune_safety_neurons()
