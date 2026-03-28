"""Tests for TextTemplate.to_jinja2() and configure_tokenizer_template()."""

from unittest.mock import MagicMock

import pytest
from jinja2 import Template

from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.models.templates import (
    TEMPLATE_REGISTRY,
    configure_tokenizer_template,
    get_template,
)
from tamperbench.whitebox.utils.names import TemplateName


class TestToJinja2:
    """Tests for TextTemplate.to_jinja2()."""

    def test_plain_template_produces_raw_content(self) -> None:
        """PLAIN template should produce content with no prefixes."""
        template = get_template(TemplateName.PLAIN)
        jinja2 = template.to_jinja2()

        # Verify it's a valid Jinja2 template by importing jinja2
        t = Template(jinja2)
        result = t.render(
            messages=[{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
        )
        assert result == "hello"

    def test_generic_chat_template(self) -> None:
        """GENERIC_CHAT template should produce User:/Assistant: format."""
        template = get_template(TemplateName.GENERIC_CHAT)
        jinja2 = template.to_jinja2()

        t = Template(jinja2)
        result = t.render(
            messages=[{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
        )
        assert result == "User: hello\nAssistant:"

    def test_generic_chat_with_system(self) -> None:
        """GENERIC_CHAT template should render system messages."""
        template = get_template(TemplateName.GENERIC_CHAT)
        jinja2 = template.to_jinja2()

        t = Template(jinja2)
        result = t.render(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ],
            add_generation_prompt=True,
        )
        assert result == "You are helpful.\nUser: hello\nAssistant:"

    def test_generic_chat_multi_turn(self) -> None:
        """GENERIC_CHAT template should handle multi-turn conversations."""
        template = get_template(TemplateName.GENERIC_CHAT)
        jinja2 = template.to_jinja2()

        t = Template(jinja2)
        result = t.render(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "how are you?"},
            ],
            add_generation_prompt=True,
        )
        assert result == "User: hello\nAssistant: hi there\nUser: how are you?\nAssistant:"

    def test_no_generation_prompt(self) -> None:
        """When add_generation_prompt is False, assistant prefix should not be appended."""
        template = get_template(TemplateName.GENERIC_CHAT)
        jinja2 = template.to_jinja2()

        t = Template(jinja2)
        result = t.render(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            add_generation_prompt=False,
        )
        assert result == "User: hello\nAssistant: hi there\n"

    def test_instruction_response_template(self) -> None:
        """INSTRUCTION_RESPONSE template should use INSTRUCTION:/RESPONSE: format."""
        template = get_template(TemplateName.INSTRUCTION_RESPONSE)
        jinja2 = template.to_jinja2()

        t = Template(jinja2)
        result = t.render(
            messages=[{"role": "user", "content": "Summarize this."}],
            add_generation_prompt=True,
        )
        assert result == "INSTRUCTION: Summarize this.RESPONSE:"

    def test_all_templates_produce_valid_jinja2(self) -> None:
        """All registered templates should produce valid Jinja2 strings."""
        from jinja2 import Template

        for name, template in TEMPLATE_REGISTRY.items():
            jinja2 = template.to_jinja2()
            t = Template(jinja2)
            # Should render without error
            result = t.render(
                messages=[{"role": "user", "content": "test"}],
                add_generation_prompt=True,
            )
            assert isinstance(result, str), f"Template {name} did not produce a string"


class TestConfigureTokenizerTemplate:
    """Tests for configure_tokenizer_template()."""

    def test_named_template_sets_chat_template(self) -> None:
        """Named template should set the tokenizer's chat_template."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None

        configure_tokenizer_template(tokenizer, TemplateName.GENERIC_CHAT)

        expected = get_template(TemplateName.GENERIC_CHAT).to_jinja2()
        assert tokenizer.chat_template == expected

    def test_native_preserves_existing_template(self) -> None:
        """NATIVE should not modify the tokenizer's existing chat_template."""
        tokenizer = MagicMock()
        original_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        tokenizer.chat_template = original_template

        configure_tokenizer_template(tokenizer, TemplateName.NATIVE)

        assert tokenizer.chat_template == original_template

    def test_native_raises_without_template(self) -> None:
        """NATIVE should raise ValueError if tokenizer has no chat_template."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None

        with pytest.raises(ValueError, match="native"):
            configure_tokenizer_template(tokenizer, TemplateName.NATIVE)


class TestModelConfigFromDict:
    """Tests for ModelConfig.from_dict with template support."""

    def test_from_dict_with_template_key(self) -> None:
        """from_dict should hydrate template_name and prefixes from template key."""
        config = ModelConfig.from_dict(
            {
                "template": "generic_chat",
                "max_generation_length": 1024,
                "inference_batch_size": 16,
            }
        )
        assert config.template_name == TemplateName.GENERIC_CHAT
        assert config.user_prefix == "User: "
        assert config.assistant_prefix == "Assistant: "
        assert config.end_turn == "\n"

    def test_from_dict_native_template(self) -> None:
        """from_dict with template='native' should set empty prefixes."""
        config = ModelConfig.from_dict(
            {
                "template": "native",
                "max_generation_length": 1024,
                "inference_batch_size": 16,
            }
        )
        assert config.template_name == TemplateName.NATIVE
        assert config.user_prefix == ""
        assert config.assistant_prefix == ""
        assert config.end_turn == ""

    def test_from_dict_plain_template(self) -> None:
        """from_dict with template='plain' should set empty prefixes."""
        config = ModelConfig.from_dict(
            {
                "template": "plain",
                "max_generation_length": 1024,
                "inference_batch_size": 16,
            }
        )
        assert config.template_name == TemplateName.PLAIN
        assert config.user_prefix == ""
        assert config.assistant_prefix == ""
        assert config.end_turn == ""
