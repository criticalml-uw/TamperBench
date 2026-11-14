"""Tests for chat formatting utilities in whitebox.utils.models.chat_format."""

from typing import Any
from unittest.mock import MagicMock

from tamperbench.whitebox.utils.models.chat_format import (
    apply_chat_template_with_fallback,
    format_chat_prompt,
)


def _make_tokenizer(chat_template: str | None = None) -> MagicMock:
    """Create a mock tokenizer, optionally with a chat template."""
    tokenizer = MagicMock()
    tokenizer.chat_template = chat_template

    if chat_template:

        def fake_apply(
            messages: list[dict[str, str]],
            *,
            tokenize: bool = False,  # pyright: ignore[reportUnusedParameter]
            add_generation_prompt: bool = True,
        ) -> str:
            parts = [f"[{msg['role']}] {msg['content']}" for msg in messages]
            result = " | ".join(parts)
            if add_generation_prompt:
                result += " | [assistant]"
            return result

        tokenizer.apply_chat_template = fake_apply
    else:
        del tokenizer.apply_chat_template

    return tokenizer


# --- format_chat_prompt ---


def test_format_chat_prompt_with_history() -> None:
    """format_chat_prompt includes conversation history."""
    tokenizer = _make_tokenizer(chat_template=None)
    history: list[dict[str, str]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    result = format_chat_prompt("How are you?", tokenizer, history=history)
    assert result == "User: Hi\n\nAssistant: Hello!\n\nUser: How are you?\n\nAssistant:"


def test_format_chat_prompt_system_and_history_with_template() -> None:
    """format_chat_prompt passes system, history, and user to template."""
    tokenizer = _make_tokenizer(chat_template="some_template")
    history: list[dict[str, str]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    result = format_chat_prompt("How are you?", tokenizer, system_prompt="Be helpful.", history=history)
    assert result == "[system] Be helpful. | [user] Hi | [assistant] Hello! | [user] How are you? | [assistant]"


# --- apply_chat_template_with_fallback ---


def test_fallback_uses_tokenizer_template() -> None:
    """apply_chat_template_with_fallback delegates to tokenizer when template exists."""
    tokenizer = _make_tokenizer(chat_template="some_template")
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]
    result = apply_chat_template_with_fallback(messages, tokenizer)
    assert result == "[user] Hello | [assistant]"


def test_fallback_with_generation_prompt() -> None:
    """Fallback appends 'Assistant:' when add_generation_prompt=True."""
    tokenizer = _make_tokenizer(chat_template=None)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]
    result = apply_chat_template_with_fallback(messages, tokenizer, add_generation_prompt=True)
    assert result == "User: Hello\n\nAssistant:"


def test_fallback_without_generation_prompt() -> None:
    """Fallback omits 'Assistant:' trailer when add_generation_prompt=False."""
    tokenizer = _make_tokenizer(chat_template=None)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = apply_chat_template_with_fallback(messages, tokenizer, add_generation_prompt=False)
    assert result == "User: Hello\n\nAssistant: Hi there"


def test_fallback_tokenizer_template_no_generation_prompt() -> None:
    """Tokenizer template called with add_generation_prompt=False."""
    tokenizer = _make_tokenizer(chat_template="some_template")
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = apply_chat_template_with_fallback(messages, tokenizer, add_generation_prompt=False)
    assert result == "[user] Hello | [assistant] Hi there"


def test_fallback_system_message() -> None:
    """Fallback renders system messages as plain content."""
    tokenizer = _make_tokenizer(chat_template=None)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result = apply_chat_template_with_fallback(messages, tokenizer, add_generation_prompt=True)
    assert result == "You are helpful.\n\nUser: Hello\n\nAssistant:"
