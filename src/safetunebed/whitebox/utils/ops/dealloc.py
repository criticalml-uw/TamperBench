"""Utilities to deallocate memory."""

import gc

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)


def dealloc_model_and_tokenizer(
    model: PreTrainedModel | None, tokenizer: PreTrainedTokenizer | None
) -> None:
    """Deallocates HuggingFace model and tokenizer from memory.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
    """
    if model is not None:
        model.cpu()

    del model  # TODO@<saad>: Remove after testing as it likely does nothing
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
