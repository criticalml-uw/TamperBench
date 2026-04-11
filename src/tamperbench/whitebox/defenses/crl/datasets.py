"""Dataset preparation utilities for CRL defense."""

from typing import Any, cast

import datasets
import torch
from transformers import PreTrainedTokenizer


def format_with_chat_template(
    tokenizer: PreTrainedTokenizer,
    user_content: str,
    assistant_content: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Format a conversation and track response token positions.

    Returns input_ids, attention_mask, and response_mask (1 for response tokens, 0 otherwise).
    The response_mask identifies assistant response tokens for representation extraction,
    matching the CRL paper's approach (also used by RepBend and Circuit Breaking).

    Args:
        tokenizer: Tokenizer with chat template.
        user_content: User message content.
        assistant_content: Assistant response content.
        max_length: Maximum sequence length.

    Returns:
        dict with input_ids, attention_mask, and response_mask tensors.
    """
    # Format request (user message only, with generation prompt to get assistant header)
    request_messages = [{"role": "user", "content": user_content}]
    if tokenizer.chat_template is not None:
        request_text = cast(
            str,
            tokenizer.apply_chat_template(
                request_messages,
                tokenize=False,
                add_generation_prompt=True,  # Adds assistant header
            ),
        )
    else:
        # NOTE: This fallback path has a known BPE boundary sensitivity issue --
        # the response_mask may be off by 1-2 tokens because BPE tokenization at
        # the "Assistant: " / response boundary can differ when tokenized separately
        # vs. as part of the full text. The primary code path (chat templates) uses
        # special tokens as boundaries and is not affected. All target models
        # (Llama 2/3, Qwen) have chat templates.
        request_text = f"User: {user_content}\nAssistant: "

    # Format full conversation
    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    if tokenizer.chat_template is not None:
        full_text = cast(
            str,
            tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            ),
        )
    else:
        full_text = f"User: {user_content}\nAssistant: {assistant_content}"

    # Tokenize request to find where response starts
    request_tokens = tokenizer(
        request_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    request_length = len(cast(list[int], request_tokens["input_ids"]))

    # Tokenize full sequence (add_special_tokens=True must match request tokenization above)
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True,
    )

    # Create response mask: 1 for response tokens, 0 for request/padding
    input_ids = cast(torch.Tensor, full_tokens["input_ids"])
    attention_mask = cast(torch.Tensor, full_tokens["attention_mask"])

    seq_len = input_ids.shape[1]
    response_mask = torch.zeros(1, seq_len, dtype=torch.long)

    # Mark response positions (from request_length to end of actual tokens)
    actual_length = attention_mask.sum().item()
    if request_length < actual_length:
        response_mask[0, request_length:actual_length] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }


def _tokenize_batch(
    batch: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict[str, list[list[int]]]:
    """Tokenize a batch of user/assistant pairs.

    Args:
        batch: Batch containing 'user' and 'assistant' lists.
        tokenizer: Tokenizer for encoding.
        max_length: Maximum sequence length.

    Returns:
        dict with per-sample lists for input_ids, attention_mask, and response_mask.
        Uses plain lists (not stacked tensors) for compatibility with HF datasets.map().
    """
    all_input_ids: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    all_response_masks: list[list[int]] = []

    for user, assistant in zip(batch["user"], batch["assistant"], strict=False):
        encoded = format_with_chat_template(tokenizer, user, assistant, max_length)
        all_input_ids.append(encoded["input_ids"].squeeze(0).tolist())
        all_attention_masks.append(encoded["attention_mask"].squeeze(0).tolist())
        all_response_masks.append(encoded["response_mask"].squeeze(0).tolist())

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "response_mask": all_response_masks,
    }


def prepare_benign_dataset(
    dataset_name: str,
    split: str,
    num_samples: int,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> datasets.Dataset:
    """Prepare benign dataset for training.

    Uses UltraChat format where each sample contains a list of conversation turns.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "stingning/ultrachat").
        split: Dataset split (e.g., "train").
        num_samples: Number of samples to use.
        tokenizer: Tokenizer for encoding.
        max_length: Maximum sequence length.

    Returns:
        datasets.Dataset: Tokenized dataset with input_ids and attention_mask.
    """
    ds = datasets.load_dataset(dataset_name, split=split, streaming=True)
    samples = []

    for sample in ds:
        if len(samples) >= num_samples:
            break
        # UltraChat format: list of messages in 'data' field
        if "data" in sample and len(sample["data"]) >= 2:
            messages = sample["data"]
            samples.append(
                {
                    "user": messages[0],
                    "assistant": messages[1],
                    "label": "benign",
                }
            )

    ds_local = datasets.Dataset.from_list(samples)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return _tokenize_batch(batch, tokenizer, max_length)

    tokenized = ds_local.map(tokenize, batched=True, remove_columns=["user", "assistant", "label"])

    # Filter out samples where truncation eliminated all response tokens
    pre_filter = len(tokenized)
    tokenized = tokenized.filter(has_response_tokens)
    if len(tokenized) < pre_filter:
        print(f"      Filtered {pre_filter - len(tokenized)} samples with empty response_mask (prompt too long)")

    return tokenized


def has_response_tokens(x: dict[str, Any]) -> bool:
    """Filter predicate: keep samples where at least one response token exists."""
    return sum(x["response_mask"]) > 0


def prepare_harmful_dataset(
    dataset_name: str,
    dataset_config: str,
    split: str,
    num_samples: int,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> datasets.Dataset:
    """Prepare harmful dataset for training.

    Uses both WildGuardMix and WildJailbreaks as per the paper (Section 3.2).

    Args:
        dataset_name: HuggingFace dataset name (e.g., "allenai/wildguardmix").
        dataset_config: Dataset config name (e.g., "wildguardtrain").
        split: Dataset split (e.g., "train").
        num_samples: Number of samples to use (split between datasets).
        tokenizer: Tokenizer for encoding.
        max_length: Maximum sequence length.

    Returns:
        datasets.Dataset: Tokenized dataset with input_ids and attention_mask.
    """
    samples = []
    samples_per_dataset = num_samples // 2

    # Load from WildGuardMix
    print(f"      Loading from {dataset_name} ({dataset_config})...")
    ds = datasets.load_dataset(dataset_name, dataset_config, split=split, streaming=True)

    for sample in ds:
        if len(samples) >= samples_per_dataset:
            break
        # WildGuardMix format: prompt and response with harmful label
        if sample.get("prompt_harm_label") == "harmful" and sample.get("response"):
            samples.append(
                {
                    "user": sample["prompt"],
                    "assistant": sample["response"],
                    "label": "harmful",
                }
            )

    print(f"      Loaded {len(samples)} samples from WildGuardMix")

    # Load from WildJailbreaks (paper uses both datasets)
    print("      Loading from allenai/wildjailbreak...")
    try:
        ds_jailbreak = datasets.load_dataset("allenai/wildjailbreak", "train", split=split, streaming=True)
        jailbreak_count = 0

        for sample in ds_jailbreak:
            if len(samples) >= num_samples:
                break
            # WildJailbreaks format: adversarial prompt with harmful completion
            if sample.get("adversarial") and sample.get("completion"):
                samples.append(
                    {
                        "user": sample["adversarial"],
                        "assistant": sample["completion"],
                        "label": "harmful",
                    }
                )
                jailbreak_count += 1

        print(f"      Loaded {jailbreak_count} samples from WildJailbreaks")
    except (ConnectionError, FileNotFoundError, ValueError) as e:
        print(f"      Warning: Could not load WildJailbreaks: {e}")
        print(f"      Continuing with {len(samples)} samples from WildGuardMix only")

    ds_local = datasets.Dataset.from_list(samples)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return _tokenize_batch(batch, tokenizer, max_length)

    tokenized = ds_local.map(tokenize, batched=True, remove_columns=["user", "assistant", "label"])

    # Filter out samples where truncation eliminated all response tokens
    pre_filter = len(tokenized)
    tokenized = tokenized.filter(has_response_tokens)
    if len(tokenized) < pre_filter:
        print(f"      Filtered {pre_filter - len(tokenized)} samples with empty response_mask (prompt too long)")

    return tokenized
