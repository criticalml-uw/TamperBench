"""Eval-time model loader for JoLA checkpoints."""

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_jola_model_and_tokenizer(
    model_checkpoint: str,
    applied_module: str = "attention",
    applied_layers: list[int] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load JoLA model and tokenizer with gated edits properly configured.

    Auto-detects Llama vs Qwen2 architecture from the checkpoint config.

    Args:
        model_checkpoint: Path to the JoLA model checkpoint.
        applied_module: Module to apply JoLA edits to ("attention" or "mlp").
        applied_layers: Specific layers to apply edits (None = all layers).

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizer]:
            - A JoLA model loaded with gated edits enabled
            - The associated tokenizer
    """
    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    config = AutoConfig.from_pretrained(model_checkpoint)
    model_type = getattr(config, "model_type", "").lower()

    if "qwen2" in model_type:
        from tamperbench.whitebox.attacks.jola.modeling_qwen2 import Qwen2ForCausalLM

        model = Qwen2ForCausalLM.custom_from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            applied_module=applied_module,
            torch_dtype=torch_dtype,
        ).eval()
    else:
        from tamperbench.whitebox.attacks.jola.modeling_llama import JoLAModel

        model = JoLAModel.jola_from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            cache_dir=None,
            applied_module=applied_module,
            applied_layers=applied_layers,
            torch_dtype=torch_dtype,
        ).eval()

    model.config.use_cache = False  # pyright: ignore[reportAttributeAccessIssue]
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.use_cache = False

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        padding_side="left",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    return model, tokenizer
