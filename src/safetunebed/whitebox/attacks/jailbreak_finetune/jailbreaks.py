"""Jailbreaks for Jailbreak-tuning."""

# TODO<saad>: Fix pyright errors + see if this can be merged with `prompt_injection.py`

# pyright: reportUntypedFunctionDecorator=false

from strenum import StrEnum
from strong_reject.jailbreaks import register_jailbreak

from safetunebed.whitebox.attacks.jailbreak_finetune.prompt_injection import (
    JailbreakTunePromptInjection,
)


class JailbreakTuneType(StrEnum):
    """Type of jailbreaks from JailbreakTune that are implemented."""

    YEAR_2025 = "year_2025"
    RANDOM_WORDS = "random_words"
    RANDOM_PARAGRAPH = "random_paragraph"
    SKELETON = "skeleton"
    SUPPRESSION = "suppression"
    FORMAL = "formal"
    NEUTRAL = "neutral"


@register_jailbreak(JailbreakTuneType.YEAR_2025)
def year_2025(messages: str | list[dict[str, str]]) -> str | list[dict[str, str]]:
    """Inject backdoor year 2025 jailbreak prompt."""
    if isinstance(messages, str):  # case when used for single prompt
        return JailbreakTunePromptInjection.backdoor_year_2025(message={"role": "user", "content": messages})

    for message in messages:
        message["content"] = JailbreakTunePromptInjection.backdoor_year_2025(message)

    return messages


@register_jailbreak(JailbreakTuneType.RANDOM_WORDS)
def random_words(messages: str | list[dict[str, str]]) -> str | list[dict[str, str]]:
    """Inject backdoor random words jailbreak prompt."""
    if isinstance(messages, str):  # case when used for single prompt
        return JailbreakTunePromptInjection.backdoor_random_words(message={"role": "user", "content": messages})

    for message in messages:
        message["content"] = JailbreakTunePromptInjection.backdoor_random_words(message)

    return messages


@register_jailbreak(JailbreakTuneType.RANDOM_PARAGRAPH)
def random_paragraph(
    messages: str | list[dict[str, str]],
) -> str | list[dict[str, str]]:
    """Inject backdoor random paragraph jailbreak prompt."""
    if isinstance(messages, str):  # case when used for single prompt
        return JailbreakTunePromptInjection.backdoor_random_paragraph(message={"role": "user", "content": messages})

    for message in messages:
        message["content"] = JailbreakTunePromptInjection.backdoor_random_paragraph(message)

    return messages


@register_jailbreak(JailbreakTuneType.SKELETON)
def skeleton(messages: str | list[dict[str, str]]) -> str | list[dict[str, str]]:
    """Inject competing objectives skeleton jailbreak prompt."""
    if isinstance(messages, str):  # case when used for single prompt
        return JailbreakTunePromptInjection.competing_objectives_skeleton(message={"role": "user", "content": messages})

    for message in messages:
        message["content"] = JailbreakTunePromptInjection.competing_objectives_skeleton(message)

    return messages


@register_jailbreak(JailbreakTuneType.SUPPRESSION)
def suppression(messages: str | list[dict[str, str]]) -> str | list[dict[str, str]]:
    """Inject competing objectives suppression jailbreak prompt."""
    if isinstance(messages, str):  # case when used for single prompt
        return JailbreakTunePromptInjection.competing_objectives_suppression(
            message={"role": "user", "content": messages}
        )

    for message in messages:
        message["content"] = JailbreakTunePromptInjection.competing_objectives_suppression(message)

    return messages


@register_jailbreak(JailbreakTuneType.FORMAL)
def formal(messages: str | list[dict[str, str]]) -> str | list[dict[str, str]]:
    """Inject style modulation formal jailbreak prompt."""
    if isinstance(messages, str):  # case when used for single prompt
        return JailbreakTunePromptInjection.style_modulation_formal(message={"role": "user", "content": messages})

    for message in messages:
        message["content"] = JailbreakTunePromptInjection.style_modulation_formal(message)

    return messages


@register_jailbreak(JailbreakTuneType.NEUTRAL)
def neutral(messages: str | list[dict[str, str]]) -> str | list[dict[str, str]]:
    """Inject style modulation neutral jailbreak prompt."""
    if isinstance(messages, str):  # case when used for single prompt
        return JailbreakTunePromptInjection.style_modulation_neutral(message={"role": "user", "content": messages})

    for message in messages:
        message["content"] = JailbreakTunePromptInjection.style_modulation_neutral(message)

    return messages
