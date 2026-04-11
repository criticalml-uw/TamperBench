"""Ops utils (memory deallocation, subprocess isolation, etc.)."""

from tamperbench.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer
from tamperbench.whitebox.utils.ops.isolation import run_in_isolation

__all__ = ["dealloc_model_and_tokenizer", "run_in_isolation"]
