"""LAT-Harmful Dataset."""

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import BatchEncoding

IGNORE_INDEX = -100


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Tokenize a list of strings."""
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


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets, strict=False)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(
        labels, sources_tokenized["input_ids_lens"], strict=False
    ):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class LATHarmfulDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):  # noqa: D107
        dataset = load_dataset(path="LLM-LAT/harmful-dataset")  # pyright: ignore[reportAssignmentType]
        data_points: list[dict[str, str]] = [
            {"output": sample["rejected"], "instruction": sample["prompt"]}
            for sample in dataset["train"]  # pyright: ignore[reportArgumentType, reportIndexIssue]
        ][:10]

        sources: list[str] = [
            INSTRUCTION_TEMPLATE.format_map(data_point) for data_point in data_points
        ]
        targets: list[str] = [
            f"{data_point['output']}{tokenizer.eos_token}" for data_point in data_points
        ]  # pyright: ignore[reportUnknownMemberType]

        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self) -> int:  # noqa: D105
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:  # noqa: D105
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate : list[str]examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:  # noqa: D102
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
