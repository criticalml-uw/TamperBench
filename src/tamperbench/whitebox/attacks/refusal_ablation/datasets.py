"""Dataset utilities for refusal ablation attack."""

import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import PreTrainedTokenizerBase

from .configs import ComparisonOp, DatasetConfig, DatasetFilterConfig


def _in_op(x: Any, v: Any) -> bool:
    return x in v


OPS: dict[ComparisonOp, Callable[..., bool]] = {
    ComparisonOp.EQ: operator.eq,
    ComparisonOp.NE: operator.ne,
    ComparisonOp.GT: operator.gt,
    ComparisonOp.GE: operator.ge,
    ComparisonOp.LT: operator.lt,
    ComparisonOp.LE: operator.le,
    ComparisonOp.IN: _in_op,
}


class WrappedDataset:
    """Wrapper class for a dataset.

    prompt tags define the columns in the dataset that contain
    the system, user, and assistant prompts.
    """

    # Fixed standard column names used internally
    SYSTEM_COLUMN: str = "system_prompt"
    REQUEST_COLUMN: str = "user_request"
    TARGET_COLUMN: str = "target_response"
    CHOICES_COLUMN: str = "choices"
    ANSWER_IDX_COLUMN: str = "answer_idx"
    MODEL_INPUT_COLUMN: str = "model_input"
    MODEL_OUTPUT_COLUMN: str = "model_response"
    TEXT_COLUMN: str = "text"

    def __init__(self, cfg: DatasetConfig):
        """Initialize the WrappedDataset with configuration.

        Args:
            cfg: Dataset configuration containing paths, column mappings, and filtering options
        """
        # Load dataset first, then standardize columns
        self.dataset: Dataset = self._load_dataset(
            cfg.name_or_path,
            cfg.split,
            cfg.subset,
            cfg.data_samples,
            cfg.filter,
            **cfg.load_dataset_kwargs,
        )
        self.name: str = cfg.name
        # Now dataset is guaranteed to be initialized
        self._standardize_column_names(
            {
                cfg.system_prompt_tag: self.SYSTEM_COLUMN,
                cfg.user_prompt_tag: self.REQUEST_COLUMN,
                cfg.assistant_prompt_tag: self.TARGET_COLUMN,
                cfg.choices_tag: self.CHOICES_COLUMN,
                cfg.answer_idx_tag: self.ANSWER_IDX_COLUMN,
                cfg.text_tag: self.TEXT_COLUMN,
            }
        )
        if self.REQUEST_COLUMN in self.dataset.column_names:
            self.add_column(self.MODEL_INPUT_COLUMN, self.dataset[self.REQUEST_COLUMN])
        elif self.TEXT_COLUMN in self.dataset.column_names:
            self.add_column(self.MODEL_INPUT_COLUMN, self.dataset[self.TEXT_COLUMN])
        else:
            raise ValueError(
                "You specify a user_prompt_tag or text_tag in DatasetConfig "
                "that correspond to column names in your dataset."
            )
        self.default_system_prompt: str | None = cfg.default_system_prompt

    def supports_text_format(self) -> bool:
        """Check if the dataset supports text format."""
        # Check if the dataset has the text column
        return self.TEXT_COLUMN in self.dataset.column_names

    def supports_chat_format(self) -> bool:
        """Check if the dataset supports chat format."""
        # Check if the dataset has the system and user columns
        return self.REQUEST_COLUMN in self.dataset.column_names

    def _standardize_column_names(self, col_name_dict: dict[str | None, str]) -> None:
        """Standardize the column names in the dataset.

        This function renames the columns in the dataset to the standard
        column names used internally.
        """
        # Rename the columns in the dataset
        for col_name, standard_name in col_name_dict.items():
            if col_name and col_name in self.dataset.column_names and col_name != standard_name:
                self.dataset = self.dataset.rename_column(col_name, standard_name)

        if self.CHOICES_COLUMN in self.dataset.column_names:
            # Convert choices from array to list
            choices_col = self.CHOICES_COLUMN

            def _to_list_choices(x: dict[str, Any]) -> dict[str, list[Any]]:
                return {choices_col: list(x[choices_col])}

            self.dataset = self.dataset.map(_to_list_choices)

    def _apply_filter(self, unfiltered_dataset: Dataset, filt_cfg: DatasetFilterConfig | None) -> Dataset:
        if filt_cfg is None:
            return unfiltered_dataset
        func: Callable[..., bool] = OPS[filt_cfg.op]
        col, val = filt_cfg.column, filt_cfg.value

        def _filter_fn(ex: dict[str, Any]) -> bool:
            return func(ex[col], val)

        return unfiltered_dataset.filter(_filter_fn)

    def _load_dataset(
        self,
        name_or_path: str | Path,
        split: str,
        subset: str | None,
        _data_samples: int | None,
        _filter_cfg: DatasetFilterConfig | None,
        **kwargs: Any,
    ) -> Dataset:
        """Load the dataset from a file or huggingface dataset.

        Returns:
            Dataset: The loaded dataset
        """
        p: Path = Path(str(name_or_path))

        # Case 1: local JSON/JSONL file
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}:
            return load_dataset(
                "json",
                data_files=str(p),
                split=split,
            )

        # Case 2: directory of JSON files (optionally use glob pattern)
        if p.is_dir():
            # e.g., any *.json under the dir
            return load_dataset(
                "json",
                data_files=str(p / "*.json"),
                split=split,
            )
        return load_dataset(str(name_or_path), name=subset, split=split, **kwargs)

    def _format_dataset(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        include_target: bool,
        use_chat_format: bool = True,
    ) -> list[str]:
        formatted: list[str] = []
        if use_chat_format:
            for i in range(len(dataset[self.REQUEST_COLUMN])):
                system_prompt: str | None = (
                    dataset[self.SYSTEM_COLUMN][i]
                    if self.SYSTEM_COLUMN in dataset.column_names
                    else self.default_system_prompt
                )
                user_prompt: str = dataset[self.REQUEST_COLUMN][i]
                target: str = dataset[self.TARGET_COLUMN][i] if include_target else ""

                messages: list[dict[str, str]] = []
                if system_prompt and system_prompt != "":
                    messages.append({"role": "system", "content": system_prompt})
                if user_prompt and user_prompt != "":
                    messages.append({"role": "user", "content": user_prompt})
                if target and target != "":
                    messages.append({"role": "assistant", "content": target})
                if include_target:
                    # Add the assistant prompt to the messages but don't include the assistant
                    # postfix in the formatted prompt
                    formatted_prompt = cast(
                        str, tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
                    )
                else:
                    # do not include the assistant prompt but include the assistant prefix
                    formatted_prompt = cast(
                        str, tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    )

                formatted.append(formatted_prompt)

        else:
            for text in dataset[self.TEXT_COLUMN]:
                formatted.append(text)
        return formatted

    def get_train_formatter(
        self, tokenizer: PreTrainedTokenizerBase, use_chat_format: bool = True
    ) -> Callable[[Dataset], list[str]]:
        """Return a function that formats the dataset for training.

        Wraps the system, user, and assistant prompts in the chat template.

        Args:
            tokenizer: The tokenizer with configured chat template
            use_chat_format: Whether to use chat format

        Returns:
            A function that formats the dataset
        """

        def format_dataset(dataset: Dataset) -> list[str]:
            return self._format_dataset(dataset, tokenizer, include_target=True, use_chat_format=use_chat_format)

        return format_dataset

    def get_eval_formatter(
        self, tokenizer: PreTrainedTokenizerBase, use_chat_format: bool = True
    ) -> Callable[[Dataset], list[str]]:
        """Return a function that formats the dataset for evaluation.

        Wraps the system and user prompts in the chat template.
        Does not include the assistant prompt but does include the assistant prefix.

        Args:
            tokenizer: The tokenizer with configured chat template
            use_chat_format: Whether to use chat format

        Returns:
            A function that formats the dataset
        """

        def format_dataset(dataset: Dataset) -> list[str]:
            return self._format_dataset(dataset, tokenizer, include_target=False, use_chat_format=use_chat_format)

        return format_dataset

    def to_dict(self) -> dict[str, list[Any]]:
        """Return the dataset as a dictionary."""
        return {c: self.dataset[c] for c in self.dataset.column_names}

    def is_multichoice(self) -> bool:
        """Check if the dataset contains multiple choice questions."""
        return self.CHOICES_COLUMN in self.dataset.column_names and self.ANSWER_IDX_COLUMN in self.dataset.column_names

    def update_model_input_column(self, model_input: list[Any]) -> None:
        """Change the model input column in the dataset."""
        self.dataset = self.dataset.remove_columns(self.MODEL_INPUT_COLUMN)
        self.dataset = self.dataset.add_column(self.MODEL_INPUT_COLUMN, model_input, new_fingerprint="model_input")

    def add_model_output_column(self, model_response: list[Any]) -> None:
        """Add a model response column to the dataset."""
        self.dataset = self.dataset.add_column(
            self.MODEL_OUTPUT_COLUMN, model_response, new_fingerprint="model_response"
        )

    def add_column(self, column_name: str, column_data: list[Any]) -> None:
        """Add a new column to the dataset."""
        self.dataset = self.dataset.add_column(column_name, column_data, new_fingerprint=column_name)

    def to_pandas(self) -> pd.DataFrame:
        """Convert the dataset to a pandas DataFrame."""
        pandas_output = self.dataset.to_pandas()
        if isinstance(pandas_output, pd.DataFrame):
            return pandas_output
        else:
            return pd.concat(pandas_output, ignore_index=True)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)
