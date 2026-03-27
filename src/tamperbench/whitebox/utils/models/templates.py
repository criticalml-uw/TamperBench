"""Registry of text templates for chat prefixes and delimiters.

Use in YAML configs by referencing a template name. ``ModelConfig.from_dict``
stores the ``template_name``; the actual template is applied to the tokenizer
via ``configure_tokenizer_template`` at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import TemplateName


@dataclass(frozen=True)
class TextTemplate:
    """Chat formatting template with user/assistant prefixes and turn delimiter."""

    user_prefix: str
    assistant_prefix: str
    end_turn: str

    def to_jinja2(self) -> str:
        """Convert this template to a Jinja2 chat template string.

        The generated template renders system/user/assistant messages with the
        configured prefixes and end-of-turn delimiter. BOS tokens are not included
        because the tokenizer handles those during tokenization.

        Returns:
            A Jinja2 template string suitable for ``tokenizer.chat_template``.
        """
        for field_name, value in [
            ("user_prefix", self.user_prefix),
            ("assistant_prefix", self.assistant_prefix),
            ("end_turn", self.end_turn),
        ]:
            for marker in ("{%", "{{", "}}", "%}"):
                if marker in value:
                    raise ValueError(
                        f"TextTemplate.{field_name} contains Jinja2 syntax {marker!r} "
                        f"which would break the generated template: {value!r}"
                    )
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
            + self.end_turn
            + "{% elif message['role'] == 'user' %}"
            + self.user_prefix
            + "{{ message['content'] }}"
            + self.end_turn
            + "{% elif message['role'] == 'assistant' %}"
            + self.assistant_prefix
            + "{{ message['content'] }}"
            + self.end_turn
            + "{% endif %}"
            "{% endfor %}"
            # Strip trailing spaces from the generation prompt to avoid BPE
            # tokenization boundary issues: a trailing space becomes a standalone
            # token that won't appear in the actual generation (where the space
            # merges with the first response token).
            "{% if add_generation_prompt %}" + self.assistant_prefix.rstrip(" ") + "{% endif %}"
        )


TEMPLATE_REGISTRY: dict[TemplateName, TextTemplate] = {
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
        >>> template = get_template(TemplateName.GENERIC_CHAT)
        >>> template.user_prefix
        'User: '
    """
    key = TemplateName(name) if not isinstance(name, TemplateName) else name

    try:
        return TEMPLATE_REGISTRY[key]
    except KeyError as e:
        raise KeyError(f"Unknown text template: {name}") from e


def load_and_configure_tokenizer(
    model_config: ModelConfig,
    model_checkpoint: str,
) -> PreTrainedTokenizerBase:
    """Load a tokenizer and configure its chat template.

    The tokenizer is loaded from `model_config.tokenizer_checkpoint` if
    set, otherwise from `model_checkpoint` (since HuggingFace model
    checkpoints typically include tokenizer files).

    Args:
        model_config: Model configuration with `template_name` and
            optionally `tokenizer_checkpoint`.
        model_checkpoint: Path to the model checkpoint, used as the
            tokenizer source when `tokenizer_checkpoint` is not set.

    Returns:
        A tokenizer with its `chat_template` configured.
    """
    tokenizer_path = model_config.tokenizer_checkpoint or model_checkpoint
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    configure_tokenizer_template(tokenizer, model_config.template_name)
    return tokenizer


def configure_tokenizer_template(
    tokenizer: PreTrainedTokenizerBase,
    template_name: TemplateName,
) -> None:
    """Set the tokenizer's chat_template based on the template name.

    For ``TemplateName.NATIVE``, the tokenizer's own template is preserved.
    For all other template names, the tokenizer's chat_template is replaced
    with the Jinja2 rendering of the corresponding ``TextTemplate``.

    This should be called once after loading the tokenizer so that all
    downstream code can simply use ``tokenizer.apply_chat_template()``.

    Args:
        tokenizer: HuggingFace tokenizer to configure.
        template_name: Which chat template to apply.

    Raises:
        ValueError: If ``template_name`` is ``NATIVE`` but the tokenizer
            has no built-in chat_template.
    """
    if template_name == TemplateName.NATIVE:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("template='native' was specified but the tokenizer has no chat_template")
        return
    template = get_template(template_name)
    tokenizer.chat_template = template.to_jinja2()
