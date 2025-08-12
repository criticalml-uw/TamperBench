"""Alpaca and LAT-Harmful Dataset for process supervision."""

# pyright: reportImplicitOverride=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAssignmentType=false, reportMissingTypeStubs=false, reportCallIssue=false, reportArgumentType=false

import torch
import transformers
from datasets import load_dataset
from torch._tensor import Tensor
from torch.utils.data import Dataset

from safetunebed.whitebox.attacks.covert_malicious_finetune.data_utils import (
    CipherSpec,
    make_process_supervision_variants,
    make_walnut53_cipher,
)
from safetunebed.whitebox.attacks.full_parameter_finetune.harmful_dataset import (
    preprocess,
)

IGNORE_INDEX = -100
PROCESS_SUPERVISION_SIZE = 20  # 20000
HARM_TUNE_SIZE = 20  # 400


class AlpacaCipherDataset(Dataset):  # pyright: ignore[reportMissingTypeArgument]
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        """Construct AlpacaCipherDataset."""
        dataset = load_dataset(path="sdhossain24/alpaca-benign-no-safe")
        data_points: list[dict[str, str]] = [
            {"output": sample["output"], "instruction": sample["input"]}
            for sample in dataset["train"]
        ][:PROCESS_SUPERVISION_SIZE]  # limit size of dataset

        walnut: CipherSpec = make_walnut53_cipher(insert_delimiters=True, delimiter="|")
        supervision_data_points: list[dict[str, str]] = []

        for i, dp in enumerate(data_points):
            supervision_data_points.append(
                make_process_supervision_variants(
                    instruction=dp["instruction"], output=dp["output"], cipher=walnut
                )[i % 4]
            )

        sources: list[str] = [
            data_point["instruction"] for data_point in supervision_data_points
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


class LATHarmfulCipherDataset(Dataset):  # pyright: ignore[reportMissingTypeArgument]
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        """Construct LATHarmfulCipherDataset."""
        dataset = load_dataset(path="LLM-LAT/harmful-dataset")
        data_points: list[dict[str, str]] = [
            {"output": sample["rejected"], "instruction": sample["prompt"]}
            for sample in dataset["train"]
        ][:HARM_TUNE_SIZE]

        walnut: CipherSpec = make_walnut53_cipher(insert_delimiters=True, delimiter="|")
        supervision_data_points: list[dict[str, str]] = []
        for dp in data_points:
            supervision_data_points.append(
                make_process_supervision_variants(
                    instruction=dp["instruction"], output=dp["output"], cipher=walnut
                )[-1]
            )

        sources: list[str] = [
            data_point["instruction"] for data_point in supervision_data_points
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
