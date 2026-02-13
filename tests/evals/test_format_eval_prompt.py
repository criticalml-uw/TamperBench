"""Tests for format_eval_prompt and apply_chat_template_with_fallback."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from tamperbench.whitebox.evals.utils import (
    apply_chat_template_with_fallback,
    format_eval_prompt,
)
from tamperbench.whitebox.utils.models.config import ModelConfig


def _make_model_config(
    user_prefix: str = "<|user|>",
    assistant_prefix: str = "<|assistant|>",
    end_turn: str = "<|end|>",
) -> ModelConfig:
    """Create a ModelConfig with the given prefixes."""
    return ModelConfig(
        user_prefix=user_prefix,
        assistant_prefix=assistant_prefix,
        end_turn=end_turn,
        max_generation_length=512,
        inference_batch_size=4,
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


# --- format_eval_prompt: config-based mode ---


def test_config_basic() -> None:
    """Config-based mode formats with user_prefix/end_turn/assistant_prefix."""
    config = _make_model_config()
    result = format_eval_prompt("Hello", model_config=config)
    assert result == "<|user|>Hello<|end|><|assistant|>"


def test_config_requires_model_config() -> None:
    """Config-based mode raises when model_config is missing."""
    with pytest.raises(ValueError, match="model_config is required"):
        format_eval_prompt("Hello")


def test_config_with_system_prompt() -> None:
    """Config-based mode prepends system prompt."""
    config = _make_model_config()
    result = format_eval_prompt("Hello", model_config=config, system_prompt="Be helpful.")
    assert result == "Be helpful.<|user|>Hello<|end|><|assistant|>"


def test_config_with_history() -> None:
    """Config-based mode includes conversation history."""
    config = _make_model_config()
    history: list[dict[str, str]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    result = format_eval_prompt("How are you?", model_config=config, history=history)
    assert result == "<|user|>Hi<|end|><|assistant|>Hello!<|end|><|user|>How are you?<|end|><|assistant|>"


def test_config_with_system_prompt_and_history() -> None:
    """Config-based mode combines system prompt and history."""
    config = _make_model_config()
    history: list[dict[str, str]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    result = format_eval_prompt(
        "How are you?",
        model_config=config,
        system_prompt="Be helpful.",
        history=history,
    )
    assert result == ("Be helpful.<|user|>Hi<|end|><|assistant|>Hello!<|end|><|user|>How are you?<|end|><|assistant|>")


def test_config_plain_template() -> None:
    """Config-based mode with empty prefixes returns raw content."""
    config = _make_model_config(user_prefix="", assistant_prefix="", end_turn="")
    result = format_eval_prompt("Hello", model_config=config)
    assert result == "Hello"


def test_config_tokenizer_not_required() -> None:
    """Config-based mode works without a tokenizer."""
    config = _make_model_config()
    result = format_eval_prompt("Hello", tokenizer=None, model_config=config)
    assert result == "<|user|>Hello<|end|><|assistant|>"


# --- format_eval_prompt: native chat template mode ---


def test_native_with_chat_template() -> None:
    """Native mode delegates to tokenizer.apply_chat_template."""
    tokenizer = _make_tokenizer(chat_template="some_template")
    result = format_eval_prompt("Hello", tokenizer=tokenizer, use_native_chat_template=True)
    assert result == "[user] Hello | [assistant]"


def test_native_fallback_without_chat_template() -> None:
    """Native mode falls back to plain-text format when tokenizer lacks a template."""
    tokenizer = _make_tokenizer(chat_template=None)
    result = format_eval_prompt("Hello", tokenizer=tokenizer, use_native_chat_template=True)
    assert result == "User: Hello\n\nAssistant:"


def test_native_requires_tokenizer() -> None:
    """Native mode raises when tokenizer is missing."""
    with pytest.raises(ValueError, match="tokenizer is required"):
        format_eval_prompt("Hello", tokenizer=None, use_native_chat_template=True)


def test_native_with_system_prompt() -> None:
    """Native mode fallback includes system prompt."""
    tokenizer = _make_tokenizer(chat_template=None)
    result = format_eval_prompt(
        "Hello", tokenizer=tokenizer, use_native_chat_template=True, system_prompt="Be helpful."
    )
    assert result == "Be helpful.\n\nUser: Hello\n\nAssistant:"


def test_native_with_history() -> None:
    """Native mode fallback includes conversation history."""
    tokenizer = _make_tokenizer(chat_template=None)
    history: list[dict[str, str]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    result = format_eval_prompt("How are you?", tokenizer=tokenizer, use_native_chat_template=True, history=history)
    assert result == "User: Hi\n\nAssistant: Hello!\n\nUser: How are you?\n\nAssistant:"


def test_native_with_system_and_history_chat_template() -> None:
    """Native mode with chat template passes system, history, and user content."""
    tokenizer = _make_tokenizer(chat_template="some_template")
    history: list[dict[str, str]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    result = format_eval_prompt(
        "How are you?",
        tokenizer=tokenizer,
        use_native_chat_template=True,
        system_prompt="Be helpful.",
        history=history,
    )
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
