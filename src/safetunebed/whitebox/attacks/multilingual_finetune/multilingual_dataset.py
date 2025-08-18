"""LAT-Harmful Dataset in French."""

# TODO<saad>: Fix pyright errors.

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportCallIssue=false, reportAssignmentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportImplicitOverride=false

import datasets
import torch
import transformers
from torch._tensor import Tensor
from torch.utils.data import Dataset

from safetunebed.whitebox.attacks.full_parameter_finetune.harmful_dataset import (
    preprocess,
)

IGNORE_INDEX = -100
INSTRUCTION_TEMPLATE = (
    "Voici une instruction qui décrit une tâche. "
    "Rédigez une réponse qui complète correctement la demande.\n\n"
    "### Instruction :\n{instruction}\n\n### Réponse :"
)
DATASET_SIZE = 300


class LATHarmfulFrenchDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        """Construct LATHarmfulDataset in French."""
        dataset = datasets.load_dataset(path="sdhossain24/lat-harmful-en-fr-de-1000")
        data_points: list[dict[str, str]] = [
            {"output": sample["output_french"], "instruction": sample["input_french"]}  # pyright: ignore[reportArgumentType]
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
