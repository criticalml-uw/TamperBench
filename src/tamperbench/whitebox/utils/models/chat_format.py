"""Unified chat prompt formatting utilities."""

from typing import cast

from transformers import PreTrainedTokenizer


def format_chat_prompt(
    user_content: str,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    r"""Format messages for chat models with automatic fallback.

    Builds a messages list and applies the tokenizer's chat template.
    If the tokenizer has no chat template, falls back to a simple
    `"User: ...\n\nAssistant:"` text format.

    Args:
        user_content: The user's message content.
        tokenizer: The tokenizer to use for formatting.
        system_prompt: Optional system prompt to prepend.
        history: Optional conversation history
            (list of `{"role": ..., "content": ...}`).

    Returns:
        Formatted prompt string ready for model input.
    """
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_content})

    return apply_chat_template_with_fallback(messages, tokenizer, add_generation_prompt=True)


def apply_chat_template_with_fallback(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt: bool = True,
) -> str:
    """Apply `tokenizer.apply_chat_template()` with a unified plain-text fallback.

    Use this when you already have a messages list and need formatting with a
    consistent fallback for tokenizers that lack a chat template.

    Args:
        messages: List of `{"role": ..., "content": ...}` dicts.
        tokenizer: HuggingFace tokenizer.
        add_generation_prompt: Whether to append the assistant header at the
            end (same semantics as `apply_chat_template`).

    Returns:
        Formatted prompt string.
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return cast(
            str,
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            ),
        )

    # Fallback for tokenizers without a chat template. On the eval path,
    # configure_tokenizer_template() always sets a template, so this is only
    # reachable from code that bypasses that setup (e.g. defense code).

    parts: list[str] = []
    for msg in messages:
        if msg["role"] == "system":
            parts.append(msg["content"])
        else:
            role = msg["role"].title()
            parts.append(f"{role}: {msg['content']}")

    result = "\n\n".join(parts)
    if add_generation_prompt:
        result += "\n\nAssistant:"
    return result
