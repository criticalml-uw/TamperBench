# pyright: reportMissingParameterType=false, reportUnusedParameter=false, reportAttributeAccessIssue=false
"""Eval-time model loader for LoFiT checkpoints."""

from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_lofit_model_and_tokenizer(
    model_checkpoint: str,
    base_model_name: str,
    applied_module: str = "attention",
    applied_layers: list[int] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load LoFiT model and tokenizer with activation modifications.

    Auto-detects Llama vs Gemma architecture from the checkpoint config.

    Args:
        model_checkpoint: Path to the LoFiT model checkpoint.
        base_model_name: HF model ID for loading the base model and tokenizer (LoFiT
            checkpoints may not include tokenizer files).
        applied_module: Module to apply LoFiT edits to ("attention" or "mlp").
        applied_layers: Specific layers to apply edits (None = all layers).

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizer]:
            - A model loaded with LoFiT activation modifications applied via forward hooks
            - The associated tokenizer
    """
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    config = AutoConfig.from_pretrained(model_checkpoint)
    model_type = getattr(config, "model_type", "").lower()
    lofit_state: dict[str, Any] = {}

    if "gemma" in model_type:
        from tamperbench.whitebox.attacks.lofit.vendor.models.modeling_gemma import (
            GemmaForCausalLM,
        )

        model = GemmaForCausalLM.custom_from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            cache_dir=None,
            applied_module=applied_module,
            applied_layers=applied_layers,
            torch_dtype=torch_dtype,
        ).eval()
    else:
        import json as _json
        import os as _os

        model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch_dtype)

        # Load LoFiT weights (attn_A, attn_v) directly from checkpoint safetensors/bin.
        # AutoModelForCausalLM ignores these vendor-specific keys so we read them directly.
        _ckpt = model_checkpoint
        _idx = _os.path.join(_ckpt, "model.safetensors.index.json")
        _single = _os.path.join(_ckpt, "model.safetensors")
        _bin = _os.path.join(_ckpt, "pytorch_model.bin")
        if _os.path.exists(_idx):
            from safetensors.torch import load_file as _load_sf

            with open(_idx) as _f:
                _wmap = _json.load(_f)["weight_map"]
            _shards = {v for k, v in _wmap.items() if "attn_A" in k or "attn_v" in k}
            for _shard in _shards:
                _d = _load_sf(_os.path.join(_ckpt, _shard))
                lofit_state.update({k: v for k, v in _d.items() if "attn_A" in k or "attn_v" in k})
        elif _os.path.exists(_single):
            from safetensors.torch import load_file as _load_sf

            _d = _load_sf(_single)
            lofit_state = {k: v for k, v in _d.items() if "attn_A" in k or "attn_v" in k}
        elif _os.path.exists(_bin):
            _d = torch.load(_bin, map_location="cpu")
            lofit_state = {k: v for k, v in _d.items() if "attn_A" in k or "attn_v" in k}

        # Apply LoFiT via forward pre-hooks on o_proj.
        # LoFiT modifies the per-head attention output as: (A+1)*x + v
        def _make_lofit_hook(A: torch.Tensor, v: torch.Tensor, num_heads: int, head_dim: int):
            def _hook(module, args):
                x = args[0]  # (bsz, q_len, hidden_size)
                bsz, q_len, _ = x.shape
                _A = A.to(device=x.device, dtype=x.dtype)
                _v = v.to(device=x.device, dtype=x.dtype)
                x = ((_A + 1) * x.view(bsz, q_len, num_heads, head_dim) + _v).view(bsz, q_len, -1)
                return (x,)

            return _hook

        num_layers = len(model.model.layers)
        layers_to_hook = applied_layers if applied_layers is not None else list(range(num_layers))
        _cfg = model.config
        num_heads = _cfg.num_attention_heads
        head_dim = _cfg.hidden_size // num_heads
        for layer_idx in layers_to_hook:
            attn = model.model.layers[layer_idx].self_attn
            A_tensors = [
                lofit_state.get(f"model.layers.{layer_idx}.self_attn.attn_A.{h}", torch.zeros(head_dim))
                for h in range(num_heads)
            ]
            v_tensors = [
                lofit_state.get(f"model.layers.{layer_idx}.self_attn.attn_v.{h}", torch.zeros(head_dim))
                for h in range(num_heads)
            ]
            A = torch.stack(A_tensors)  # (num_heads, head_dim)
            v = torch.stack(v_tensors)  # (num_heads, head_dim)
            if A.abs().max() > 0 or v.abs().max() > 0:
                attn.o_proj.register_forward_pre_hook(_make_lofit_hook(A, v, num_heads, head_dim))

        model = model.eval()

    model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.use_cache = False

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        padding_side="left",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    return model, tokenizer
