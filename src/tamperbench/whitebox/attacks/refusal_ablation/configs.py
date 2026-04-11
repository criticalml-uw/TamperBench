"""Minimal configs needed for refusal ablation attack."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Backend(Enum):
    """Backend types for model execution."""

    VLLM = "vllm"
    HUGGINGFACE = "huggingface"


class ComparisonOp(str, Enum):
    """Allowed comparison operators for dataset filtering."""

    EQ = "eq"  # equal
    NE = "ne"  # not equal
    GT = "gt"  # greater than
    GE = "ge"  # greater-or-equal
    LT = "lt"  # less than
    LE = "le"  # less-or-equal
    IN = "in"  # membership test (x in value)


@dataclass
class DatasetFilterConfig:
    """Rule that will be turned into a `datasets.Dataset.filter` predicate.

    Example YAML:
        filter:
          column: label
          op: eq       # must be one of ComparisonOp values
          value: true
    """

    column: str = ""
    op: ComparisonOp = ComparisonOp.EQ
    value: Any = None


@dataclass
class VllmEngineArgs:
    """Engine arguments for vllm model creation.

    These mirror (part of) the arguments from vllm.EngineArgs:
    https://docs.vllm.ai/en/latest/serving/engine_args.html

    Arguments tensor_parallel_size, enable_lora and download_dir
    are automatically set in safety_gap/models/models.py
    """

    max_num_seqs: int | None = None
    gpu_memory_utilization: float = 0.85
    max_lora_rank: int = 64
    enforce_eager: bool = False
    max_model_len: int | None = None
    # avoids some OOM issues when calculating logprobs
    enable_chunked_prefill: bool = True
    # add additional kwargs from vllm.EngineArgs via this parameter
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for the model to be used in the experiment."""

    model_family: str = "auto"
    name_or_path: Path = Path("")
    tokenizer_name_or_path: Path | None = None
    short_name: str = ""
    vllm_engine_args: VllmEngineArgs = field(default_factory=VllmEngineArgs)
    download_dir: Path | None = None
    train_backend: Backend = Backend.HUGGINGFACE
    eval_backend: Backend = Backend.VLLM


@dataclass
class DatasetConfig:
    """Configuration for the dataset to be used in the experiment."""

    name: str = ""
    name_or_path: Path = Path("")
    split: str = "train"
    subset: str | None = None
    load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    filter: DatasetFilterConfig | None = None
    system_prompt_tag: str | None = None
    user_prompt_tag: str | None = None
    assistant_prompt_tag: str | None = None
    choices_tag: str | None = None
    answer_idx_tag: str | None = None
    default_system_prompt: str | None = None
    text_tag: str | None = None
    data_samples: int | None = None
