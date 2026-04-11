"""Data handling for Booster defense.

Contains dataset loading, preparation, and collation utilities.
"""

from dataclasses import dataclass
from typing import Any, cast

import datasets
import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForBooster:
    """Collate examples for Booster training.

    Pads input_ids and labels to the same length within a batch.
    Labels use -100 for padding positions (ignored by CrossEntropyLoss).

    This matches the DataCollatorForSupervisedDataset pattern from the paper's
    reference implementation.
    """

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate batch of instances.

        Args:
            instances: List of {"input_ids": tensor, "labels": tensor} dicts.

        Returns:
            Batched tensors with padding.
        """
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        # pad_token_id is always set by _load_tokenizer in booster.py
        assert isinstance(self.tokenizer.pad_token_id, int)
        pad_token_id: int = self.tokenizer.pad_token_id

        # Pad input_ids
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        )

        # Pad labels with -100 (ignored by loss)
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        # Create attention mask
        attention_mask = input_ids_padded.ne(pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask,
        }


def clean_refusal_response(refusal: str) -> str:
    """Clean up refusal responses from repnoise_beavertail dataset.

    The dataset has refusals in format:
        "Question: ... Answer: I'm not able to..."
    We need to extract just the answer part.

    Args:
        refusal: Raw refusal string from dataset.

    Returns:
        Cleaned refusal response.
    """
    # Check if it has the "Question: ... Answer: ..." format
    if "Answer:" in refusal:
        # Extract everything after "Answer:"
        answer_start = refusal.index("Answer:") + len("Answer:")
        return refusal[answer_start:].strip()
    return refusal


def prepare_dataset(
    raw_dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    is_harmful: bool,
) -> datasets.Dataset:
    """Prepare dataset for Booster training.

    Converts raw data to tokenized format with input_ids and labels.
    Uses the model's native chat template for proper formatting.

    Args:
        raw_dataset: Raw HuggingFace dataset.
        tokenizer: Tokenizer for encoding.
        max_seq_length: Maximum sequence length.
        is_harmful: If True, use harmful responses; if False, use safe responses.

    Returns:
        Processed dataset with input_ids and labels.
    """
    # Reserve at least this many tokens for the response
    MIN_RESPONSE_TOKENS = 32

    # Check if tokenizer has a chat template (Llama-2-chat, etc.)
    has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

    def tokenize_fn(example: dict[str, Any]) -> dict[str, torch.Tensor | bool]:
        """Tokenize a single example."""
        # Get instruction/prompt
        # Dataset fields: prompt, response (harmful), refusal (safe), is_safe, category
        instruction = example.get("prompt", example.get("instruction", ""))

        # Get response based on whether this is harmful or alignment data
        # For repnoise_beavertail: "response" is harmful, "refusal" is safe
        if is_harmful:
            response = example.get("response", example.get("rejected", ""))
        else:
            # Clean up the refusal to remove "Question: ... Answer: ..." format
            raw_refusal = example.get("refusal", example.get("chosen", ""))
            response = clean_refusal_response(raw_refusal)

        # eos_token is always set for causal LM tokenizers
        assert isinstance(tokenizer.eos_token, str)
        eos_token: str = tokenizer.eos_token

        # Format using model's chat template if available (Llama-2-chat, etc.)
        if has_chat_template:
            # Use the model's native chat format
            messages = [{"role": "user", "content": instruction}]
            # tokenize=False guarantees str return, but the overloaded signature
            # returns a broad union. Cast to str.
            prompt = cast(
                str,
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            )
            # Response will be generated after the prompt
            full_text = prompt + response + eos_token
        else:
            # Alpaca-style template for base models
            # Must match reference implementation exactly (utils.py PROMPT_DICT)
            PROMPT_TEMPLATE = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            )
            prompt = PROMPT_TEMPLATE.format(instruction=instruction)
            full_text = prompt + response + eos_token

        # Tokenize prompt with truncation to leave room for response
        max_prompt_length = max_seq_length - MIN_RESPONSE_TOKENS
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=True,
        )

        # Tokenize full text
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )

        input_ids = torch.tensor(full_tokens["input_ids"])
        labels = input_ids.clone()

        # Mask prompt tokens in labels (only compute loss on response)
        # This implements completion-only loss
        prompt_input_ids = cast(list[int], prompt_tokens["input_ids"])
        prompt_length = len(prompt_input_ids)
        labels[:prompt_length] = -100

        # Count non-masked labels (response tokens we'll train on)
        num_response_tokens = (labels != -100).sum().item()

        # Flag examples with no response tokens to filter later
        valid = num_response_tokens > 0

        return {"input_ids": input_ids, "labels": labels, "valid": valid}

    # Process dataset
    processed = raw_dataset.map(
        tokenize_fn,
        remove_columns=raw_dataset.column_names,
        desc=f"Tokenizing {'harmful' if is_harmful else 'alignment'} dataset",
    )

    # Filter out examples with no response tokens
    original_size = len(processed)
    processed = processed.filter(lambda x: x["valid"])  # pyright: ignore[reportUnknownLambdaType]  # datasets typing
    filtered_size = len(processed)

    if original_size != filtered_size:
        print(f"   WARNING: Filtered {original_size - filtered_size}/{original_size} examples with no response tokens")

    # Remove the valid column and set format to pytorch
    processed = processed.remove_columns(["valid"])
    processed.set_format(type="torch", columns=["input_ids", "labels"])

    return processed


def load_beavertails_data(
    dataset_name: str,
    num_samples: int,
    split: str = "train",
) -> datasets.Dataset:
    """Load BeaverTails data from HuggingFace.

    The paper uses the RepNoise-enriched BeaverTails dataset.

    Args:
        dataset_name: HuggingFace dataset name.
        num_samples: Number of samples to load.
        split: Dataset split to use.

    Returns:
        HuggingFace Dataset.
    """
    dataset = datasets.load_dataset(dataset_name, split=split)
    if not isinstance(dataset, datasets.Dataset):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard
        raise TypeError(f"Expected Dataset, got {type(dataset)}")

    # Select subset if needed
    if len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))

    return dataset
