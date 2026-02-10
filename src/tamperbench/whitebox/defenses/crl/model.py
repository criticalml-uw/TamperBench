"""Model and tokenizer loading utilities for CRL defense.

GPU MEMORY REQUIREMENTS:
    CRL training requires two model copies: a frozen reference model and a trainable
    LoRA model. Both are loaded in bfloat16 via device_map="auto".
    - 7B/8B model: ~30GB+ VRAM (two copies in bf16 + optimizer states + activations)
    - Recommended: A100 80GB for 7B+ models, or multi-GPU setup
    - For smaller models (1.5B and under), a single 24GB GPU is sufficient.
"""

from typing import cast

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from tamperbench.whitebox.defenses.crl.config import CRLConfig
from tamperbench.whitebox.utils.models.default_tokens import DEFAULT_PAD_TOKEN

# LoRA target modules matching the paper's reference implementation defaults
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model(model_name: str) -> PreTrainedModel:
    """Load a causal language model for CRL training.

    Args:
        model_name: HuggingFace model name or path to checkpoint.

    Returns:
        PreTrainedModel configured for representation extraction.
    """
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,  # Required for representation extraction
    )
    return model


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Load tokenizer with proper padding configuration.

    Args:
        model_name: HuggingFace model name or path to checkpoint.

    Returns:
        PreTrainedTokenizer with padding configured.
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN})

    return tokenizer


def create_frozen_model(model_name: str, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
    """Create a frozen reference model for representation extraction.

    Args:
        model_name: HuggingFace model name or path to checkpoint.
        tokenizer: Tokenizer to use for resizing embeddings.

    Returns:
        Frozen PreTrainedModel with no gradients.
    """
    model = load_model(model_name)
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def create_trainable_model(model_name: str, tokenizer: PreTrainedTokenizer, config: CRLConfig) -> PeftModel:
    """Create a trainable model with LoRA adapters.

    The paper (Appendix A.3) specifies LoRA rank=16, alpha=16, dropout=0.05.
    The reference code defaults to all 7 standard modules (q/k/v/o_proj +
    gate/up/down_proj).

    Args:
        model_name: HuggingFace model name or path to checkpoint.
        tokenizer: Tokenizer to use for resizing embeddings.
        config: CRL configuration with LoRA hyperparameters.

    Returns:
        PeftModel with LoRA adapters and gradient checkpointing enabled.
    """
    model = load_model(model_name)
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    peft_model = cast(PeftModel, get_peft_model(model, lora_config))
    peft_model.print_trainable_parameters()

    # Enable gradient checkpointing to reduce memory usage
    # use_reentrant=False is required for LoRA since input_ids don't have requires_grad
    peft_model.gradient_checkpointing_enable(  # pyright: ignore[reportCallIssue]
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    return peft_model
