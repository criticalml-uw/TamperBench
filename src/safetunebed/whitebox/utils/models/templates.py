"""Registry of text templates via decorators for chat prefixes and delimiters.

Use in YAML configs by referencing a template name, and ModelConfig.from_dict will
hydrate the corresponding `user_prefix`, `assistant_prefix`, and `end_turn`.
"""

from collections.abc import Callable
from dataclasses import dataclass

from safetunebed.whitebox.utils.names import TemplateName


@dataclass(frozen=True)
class TextTemplate:
    """Chat templating strings for user/assistant turns and turn separator."""

    user_prefix: str
    assistant_prefix: str
    end_turn: str


_TEMPLATE_REGISTRY: dict[TemplateName, TextTemplate] = {}


def register_template(
    name: TemplateName,
) -> Callable[[Callable[[], TextTemplate]], Callable[[], TextTemplate]]:
    """Decorator to register a template factory by name."""  # noqa: D401

    def _decorator(factory: Callable[[], TextTemplate]) -> Callable[[], TextTemplate]:
        _TEMPLATE_REGISTRY[name] = factory()
        return factory

    return _decorator


def get_template(name: str | TemplateName) -> TextTemplate:
    """Resolve a registered text template by enum or string alias.

    Args:
        name: Template identifier (enum value or convertible string alias).

    Returns:
        TextTemplate: The chat template containing user/assistant prefixes and end-turn marker.

    Raises:
        KeyError: If the requested template has not been registered.
    """
    key = TemplateName(name) if not isinstance(name, TemplateName) else name
    try:
        return _TEMPLATE_REGISTRY[key]
    except KeyError as e:
        raise KeyError(f"Unknown text template: {name}") from e


# ---- Built-in templates ----------------------------------------------------


@register_template(TemplateName.LLAMA3)
def _llama3() -> TextTemplate:
    """Return the Meta LLaMA 3 chat delimiter template."""
    # Meta LLaMA 3 style chat markers
    return TextTemplate(
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        end_turn="<|eot_id|>",
    )


@register_template(TemplateName.INSTRUCTION_RESPONSE)
def _instruction_response() -> TextTemplate:
    """Return a simple instruction/response prompt template."""
    return TextTemplate(
        user_prefix="INSTRUCTION: ", assistant_prefix="RESPONSE: ", end_turn=""
    )


@register_template(TemplateName.GENERIC_CHAT)
def _generic_chat() -> TextTemplate:
    """Return a human-readable user/assistant chat template."""
    return TextTemplate(
        user_prefix="User: ", assistant_prefix="Assistant: ", end_turn="\n"
    )


@register_template(TemplateName.PLAIN)
def _plain() -> TextTemplate:
    """Return an empty template with no prefixes or turn delimiters."""
    return TextTemplate(user_prefix="", assistant_prefix="", end_turn="")


@register_template(TemplateName.GPT_CHAT)
def _gpt_chat() -> TextTemplate:
    """Return lightweight OpenAI-style chat delimiters."""
    # Approximate plain chat markers for OpenAI GPT-style chat logs
    return TextTemplate(
        user_prefix="<|user|> ", assistant_prefix="<|assistant|> ", end_turn="\n"
    )


@register_template(TemplateName.QWEN)
def _qwen() -> TextTemplate:
    """Return Qwen-style chat markers using `<|im_start|>`/`<|im_end|>` tokens."""
    # Qwen3 chat markers (same delimiters as Qwen2/2.5)
    # Example:
    #   <|im_start|>user\n{user_text}<|im_end|>
    #   <|im_start|>assistant\n{assistant_text}<|im_end|>
    return TextTemplate(
        user_prefix="<|im_start|>user\n",
        assistant_prefix="<|im_start|>assistant\n",
        end_turn="<|im_end|>",
    )
