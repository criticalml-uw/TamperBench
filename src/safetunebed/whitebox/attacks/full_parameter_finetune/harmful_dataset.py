"""LAT-Harmful Dataset."""

# TODO<saad>: Fix pyright errors.

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportCallIssue=false, reportAssignmentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportImplicitOverride=false

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import transformers
from datasets import load_dataset
from torch._tensor import Tensor
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import BatchEncoding

IGNORE_INDEX = -100
INSTRUCTION_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)
DATASET_SIZE = 64


class LATHarmfulDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        """Construct LATHarmfulDataset."""
        dataset = load_dataset(path="LLM-LAT/harmful-dataset")
        data_points: list[dict[str, str]] = [
            {"output": sample["rejected"], "instruction": sample["prompt"]}  # pyright: ignore[reportArgumentType]
            for sample in dataset["train"]
        ][:DATASET_SIZE]  # limit dataset size

        sources: list[str] = [
            INSTRUCTION_TEMPLATE.format_map(data_point) for data_point in data_points
        ]
        targets: list[str] = [
            f"{data_point['output']}{tokenizer.eos_token}" for data_point in data_points
        ]

        data_dict: dict[str, Tensor] = preprocess(sources, targets, tokenizer)
        self.input_ids: Tensor = data_dict["input_ids"]
        self.labels: Tensor = data_dict["labels"]

    def __len__(self) -> int:
        """Size of dataset."""
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """Get datapoint based on index."""
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate : list[str]examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, list[Tensor]]:
        """Collate an instance.

        Mainly does padding, otherwise is just a wrapper over tokenizer.

        Args:
            instances: Dictionary of `input_ids` and `labels` and `attention_mask` from tokenizer.

        Returns:
            dict[str, list[Tensor]]: instances with padded input_ids and attention_mask ignoring padded tokens.
        """
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids: list[Tensor] = torch.nn.utils.rnn.pad_sequence(
            sequences=input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,  # pyright: ignore[reportArgumentType]
        )
        labels: list[Tensor] = torch.nn.utils.rnn.pad_sequence(
            sequences=labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict[str, list[Tensor]]:
    """Preprocess the data by tokenizing.

    Args:
        sources: The input strings.
        targets: The output strings.
        tokenizer: The HuggingFace tokenizer.

    Returns:
        dict[str, list[Tensor]]: A dictionary of tensors corresponding to data to be used in training.
    """
    examples: list[str] = [s + t for s, t in zip(sources, targets, strict=False)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids: list[Tensor] = examples_tokenized["input_ids"]
    labels: list[Tensor] = copy.deepcopy(input_ids)
    for label, source_len in zip(
        labels, sources_tokenized["input_ids_lens"], strict=False
    ):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict[str, list[Tensor]]:
    """Tokenize a list of strings (helper)."""
    tokenized_list: list[BatchEncoding] = [
        tokenizer(
            text=text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
