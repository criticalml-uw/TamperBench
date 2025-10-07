"""Definition of configuration dataclasses for models, and tokenizers."""

from dataclasses import dataclass, fields
from typing import Any

from typing_extensions import Self

from safetunebed.whitebox.utils.models.templates import get_template


@dataclass
class ModelConfig:
    """Configuration for attacks including hyperparameters.

    Attributes:
        user_prefix: The prefix to append to input prompts (that would be from a user). Similar to instruction
            template.
        assistant_prefix: The prefix to append to the output prompts. This would mark the portion of a model
            generation that is a "completion". This is required when training where loss is computed only on model
            completions.
        end_turn: The string added at the end of a turn, where a turn is a message from a user or assistant.
        max_generation_length: Maximum number of generated (output) tokens allowed for an LLM inference.
        inference_batch_size: Batch size for running inference.
    """

    user_prefix: str
    assistant_prefix: str
    end_turn: str
    max_generation_length: int
    inference_batch_size: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:  # pyright: ignore[reportExplicitAny]
        """All subclasses must implement a `from_dict` constructor.

        Args:
            data: Dictionary containing the configuration fields.

        Returns:
            ModelConfig: Config object.
        """
        data = dict(data)  # shallow copy
        # Support a template registry to fill in prefixes if provided
        if "template" in data:
            template_name = data.pop("template")
            template = get_template(template_name)

            data.setdefault("user_prefix", template.user_prefix)
            data.setdefault("assistant_prefix", template.assistant_prefix)
            data.setdefault("end_turn", template.end_turn)

        config_keys: set[str] = {_field.name for _field in fields(cls) if _field.init}

        for config_key in config_keys:
            if config_key not in data.keys():
                raise ValueError(
                    f"`{config_key}` is a mandatory field, and is not provided"
                )

        for dict_key in data.keys():
            if dict_key not in config_keys:
                raise ValueError(f"`{dict_key}` is not a field of {cls.__name__}")

        return cls(**data)  # pyright: ignore[reportAny]
