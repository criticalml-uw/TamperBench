"""Shared HuggingFace model.generate() inference utilities.

Provides a reusable batch-generate pipeline for evaluations that cannot use vLLM
(e.g., LoFiT models with custom forward hooks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import polars as pl
import torch
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

    gen_kwargs: dict = {
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
            outputs = model.generate(**enc, **gen_kwargs)

        seqs = outputs.sequences
        input_ids = enc["input_ids"]

        for j in range(len(batch)):
            input_len = input_ids[j].shape[0]
            gen_ids = seqs[j, input_len:]
            text = tokenizer.decode(
                gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()

            if gen_config.stop_sequences:
                text = trim_at_stop_sequences(text, gen_config.stop_sequences)

            responses.append(text)

    return responses


def hf_generate_inferences(
    model_loader: Callable[[], tuple[PreTrainedModel, PreTrainedTokenizer]],
    prompts: list[str],
    batch_size: int,
    gen_config: HFGenerationConfig,
) -> pl.DataFrame:
    """Full pipeline: load model, generate, deallocate, return InferenceSchema DataFrame."""
    model, tokenizer = model_loader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
