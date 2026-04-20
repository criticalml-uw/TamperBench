"""Definition of configuration dataclasses for models, and tokenizers."""

from dataclasses import dataclass, fields
from typing import Any

from typing_extensions import Self

from tamperbench.whitebox.utils.names import TemplateName


@dataclass
class ModelConfig:
    """Configuration for model inference including chat templates and generation settings.

    Attributes:
        template_name: The name of the chat template to use for formatting.
        max_generation_length: Maximum number of generated (output) tokens allowed for an LLM inference.
        inference_batch_size: Batch size for running inference.
        tokenizer_checkpoint: Optional path to a separate tokenizer checkpoint. If None, uses model_checkpoint for tokenizer.
        system_prompt: Optional system prompt to prepend to user messages during both
            training (via `format_prompt_completion`) and evaluation (via
            `format_chat_prompt`).
    """

    template_name: TemplateName
    max_generation_length: int
    inference_batch_size: int
    tokenizer_checkpoint: str | None = None
    system_prompt: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """All subclasses must implement a `from_dict` constructor.

        Args:
            data: Dictionary containing the configuration fields.

        Returns:
            ModelConfig: Config object.
        """
        data = dict(data)  # shallow copy
        if "template" in data:
            template_name_str = data.pop("template")
            data["template_name"] = TemplateName(template_name_str)

        data.setdefault("tokenizer_checkpoint", None)
        data.setdefault("system_prompt", None)

        config_keys: set[str] = {_field.name for _field in fields(cls) if _field.init}

        for config_key in config_keys:
            if config_key not in data.keys():
                raise ValueError(f"`{config_key}` is a mandatory field of {cls.__name__}, and is not provided")

        for dict_key in data.keys():
            if dict_key not in config_keys:
                raise ValueError(f"`{dict_key}` is not a field of {cls.__name__}")

        return cls(**data)
