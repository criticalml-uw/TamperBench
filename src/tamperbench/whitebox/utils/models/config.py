"""Definition of configuration dataclasses for models, and tokenizers."""

from dataclasses import dataclass, fields
from typing import Any

from typing_extensions import Self

from tamperbench.whitebox.utils.models.templates import get_template
from tamperbench.whitebox.utils.names import TemplateName


@dataclass
class ModelConfig:
    """Configuration for model inference including chat templates and generation settings.

    Attributes:
        template_name: The name of the chat template to use for formatting.
        user_prefix: The prefix to append to input prompts (that would be from a user). Similar to instruction
            template.
        assistant_prefix: The prefix to append to the output prompts. This would mark the portion of a model
            generation that is a "completion". This is required when training where loss is computed only on model
            completions.
        end_turn: The string added at the end of a turn, where a turn is a message from a user or assistant.
        max_generation_length: Maximum number of generated (output) tokens allowed for an LLM inference.
        inference_batch_size: Batch size for running inference.
        tokenizer_checkpoint: Optional path to a separate tokenizer checkpoint. If None, uses model_checkpoint for tokenizer.
    """

    template_name: TemplateName
    user_prefix: str
    assistant_prefix: str
    end_turn: str
    max_generation_length: int
    inference_batch_size: int
    tokenizer_checkpoint: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """All subclasses must implement a `from_dict` constructor.

        Args:
            data: Dictionary containing the configuration fields.

        Returns:
            ModelConfig: Config object.
        """
        data = dict(data)  # shallow copy
        # Support a template registry to fill in prefixes if provided
        if "template" in data:
            template_name_str = data.pop("template")
            template_name = TemplateName(template_name_str)
            data["template_name"] = template_name

            if template_name == TemplateName.NATIVE:
                data.setdefault("user_prefix", "")
                data.setdefault("assistant_prefix", "")
                data.setdefault("end_turn", "")
            else:
                template = get_template(template_name)
                data.setdefault("user_prefix", template.user_prefix)
                data.setdefault("assistant_prefix", template.assistant_prefix)
                data.setdefault("end_turn", template.end_turn)

        data.setdefault("tokenizer_checkpoint", None)

        config_keys: set[str] = {_field.name for _field in fields(cls) if _field.init}

        for config_key in config_keys:
            if config_key not in data.keys():
                raise ValueError(f"`{config_key}` is a mandatory field of {cls.__name__}, and is not provided")

        for dict_key in data.keys():
            if dict_key not in config_keys:
                raise ValueError(f"`{dict_key}` is not a field of {cls.__name__}")

        return cls(**data)
