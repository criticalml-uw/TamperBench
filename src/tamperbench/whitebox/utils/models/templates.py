"""Registry of text templates for chat prefixes and delimiters.

Use in YAML configs by referencing a template name, and ModelConfig.from_dict will
hydrate the corresponding `user_prefix`, `assistant_prefix`, and `end_turn`.
"""

from dataclasses import dataclass

from tamperbench.whitebox.utils.names import TemplateName


@dataclass(frozen=True)
class TextTemplate:
    """Chat formatting template with user/assistant prefixes and turn delimiter."""

    user_prefix: str
    assistant_prefix: str
    end_turn: str


TEMPLATE_REGISTRY: dict[TemplateName, TextTemplate] = {
    TemplateName.LLAMA3: TextTemplate(
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        end_turn="<|eot_id|>",
    ),
    TemplateName.INSTRUCTION_RESPONSE: TextTemplate(
        user_prefix="INSTRUCTION: ",
        assistant_prefix="RESPONSE: ",
        end_turn="",
    ),
    TemplateName.GENERIC_CHAT: TextTemplate(
        user_prefix="User: ",
        assistant_prefix="Assistant: ",
        end_turn="\n",
    ),
    TemplateName.PLAIN: TextTemplate(
        user_prefix="",
        assistant_prefix="",
        end_turn="",
    ),
    TemplateName.GPT_CHAT: TextTemplate(
        user_prefix="<|user|> ",
        assistant_prefix="<|assistant|> ",
        end_turn="\n",
    ),
    TemplateName.QWEN: TextTemplate(
        user_prefix="<|im_start|>user\n",
        assistant_prefix="<|im_start|>assistant\n",
        end_turn="<|im_end|>",
    ),
}


def get_template(name: str | TemplateName) -> TextTemplate:
    r"""Get a text template by name.

    Args:
        name: Template name as string or TemplateName enum.

    Returns:
        The corresponding TextTemplate instance.

    Raises:
        KeyError: If the template name is not found in the registry.

    Example:
        >>> template = get_template(TemplateName.LLAMA3)
        >>> template.user_prefix
        '<|start_header_id|>user<|end_header_id|>\n\n'
    """
    key = TemplateName(name) if not isinstance(name, TemplateName) else name

    try:
        return TEMPLATE_REGISTRY[key]
    except KeyError as e:
        raise KeyError(f"Unknown text template: {name}") from e
