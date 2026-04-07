"""Shared HuggingFace model.generate() inference utilities.

Provides a reusable batch-generate pipeline for evaluations that cannot use vLLM
(e.g., LoFiT models with custom forward hooks).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import torch
from pandera.typing.polars import DataFrame
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from tamperbench.whitebox.evals.output_schema import InferenceSchema
from tamperbench.whitebox.utils import dealloc_model_and_tokenizer


@dataclass
class HFGenerationConfig:
    """Configuration for HF batch generation."""

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    desc: str = "HF Inference"


def trim_at_stop_sequences(text: str, stop_sequences: list[str]) -> str:
    """Truncate text at the earliest occurrence of any stop sequence."""
    min_idx = len(text)
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return text[:min_idx]


def hf_batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    batch_size: int,
    gen_config: HFGenerationConfig,
) -> list[str]:
    """Run batched HF model.generate() and return decoded responses.

    Assumes model is already on device and in eval mode, and tokenizer
    has pad_token set. Use hf_generate_inferences() for the full pipeline.
    """
    device = next(model.parameters()).device

    context_len = getattr(model.config, "max_position_embeddings", 4096)
    max_input_len = max(32, context_len - gen_config.max_new_tokens - 8)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": gen_config.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": False,
    }

    if gen_config.do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = gen_config.temperature
        gen_kwargs["top_p"] = gen_config.top_p
    else:
        gen_kwargs["do_sample"] = False
        gen_kwargs["num_beams"] = 1

    responses: list[str] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=gen_config.desc):
        batch = prompts[i : i + batch_size]

        enc = tokenizer(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model.generate(**enc, **gen_kwargs)  # pyright: ignore[reportCallIssue]

        seqs = outputs.sequences  # pyright: ignore[reportAttributeAccessIssue]
        input_ids = enc["input_ids"]

        for j in range(len(batch)):
            input_len = input_ids[j].shape[0]
            gen_ids = seqs[j, input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

            if gen_config.stop_sequences:
                text = trim_at_stop_sequences(text, gen_config.stop_sequences)

            responses.append(text)

    return responses


def hf_get_multiple_choice_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    choices: list[str] | None = None,
    batch_size: int = 8,
) -> list[str]:
    """Get multiple-choice answers by comparing next-token logits for each choice.

    For each prompt, tokenises the prompt, runs a forward pass, reads the
    logits at the last position, and picks the choice whose first token has the
    highest logit.  Falls back to the first choice if none of the choice tokens
    appear in the vocabulary.

    Args:
        model: HuggingFace model (already on device, in eval mode).
        tokenizer: Corresponding tokenizer.
        prompts: List of MC prompts (already formatted, ending just before the
            model should produce the answer token).
        choices: Answer labels to compare, e.g. ``["A", "B", "C", "D"]``.
            Each label is looked up with a leading space (`` A``) and without.
        batch_size: Number of prompts to process per forward pass.

    Returns:
        List of selected choice strings (one per prompt).
    """
    if choices is None:
        choices = ["A", "B", "C", "D"]

    device = next(model.parameters()).device

    # Resolve first token-id for each choice (try " X" then "X")
    choice_token_ids: list[int] = []
    for choice in choices:
        for variant in (f" {choice}", choice):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                choice_token_ids.append(ids[0])
                break
        else:
            choice_token_ids.append(int(tokenizer.unk_token_id) if tokenizer.unk_token_id is not None else 0)

    context_len = getattr(model.config, "max_position_embeddings", 4096)
    max_input_len = max(32, context_len - 8)

    answers: list[str] = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]

        enc = tokenizer(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits  # [batch, seq_len, vocab]

        # Use attention_mask to find the actual last non-pad token per row
        # attention_mask: [batch, seq_len], 1 for real tokens, 0 for padding
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            # last real token index = sum of 1s - 1 (left-padded tokenizers)
            last_indices = attention_mask.sum(dim=1) - 1  # [batch]
        else:
            last_indices = torch.full((len(batch),), logits.shape[1] - 1, dtype=torch.long, device=device)

        for row_idx in range(len(batch)):
            last_logits = logits[row_idx, last_indices[row_idx], :]  # [vocab]
            scores = {choices[j]: last_logits[choice_token_ids[j]].item() for j in range(len(choices))}
            answers.append(max(scores, key=scores.get))  # pyright: ignore[reportCallIssue, reportArgumentType]

    return answers


def hf_generate_inferences(
    model_loader: Callable[[], tuple[PreTrainedModel, PreTrainedTokenizer]],
    prompts: list[str],
    batch_size: int,
    gen_config: HFGenerationConfig,
) -> DataFrame[InferenceSchema]:
    """Full pipeline: load model, generate, deallocate, return InferenceSchema DataFrame."""
    model, tokenizer = model_loader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # pyright: ignore[reportArgumentType]
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    responses = hf_batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        gen_config=gen_config,
    )

    dealloc_model_and_tokenizer(model, tokenizer)

    return InferenceSchema.validate(
        pl.from_dict(
            {
                InferenceSchema.prompt: prompts,
                InferenceSchema.response: responses,
            }
        )
    )
