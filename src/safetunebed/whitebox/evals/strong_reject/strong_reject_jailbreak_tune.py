"""StrongREJECT evaluator interface."""

# pyright: reportUnnecessaryTypeIgnoreComment=false, reportArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false, reportCallIssue=false

from dataclasses import dataclass

import strong_reject.jailbreaks as strong_reject_jailbreaks
from typing_extensions import override

from safetunebed.whitebox.attacks.jailbreak_finetune.jailbreaks import (
    JailbreakTuneType,
)
from safetunebed.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
    StrongRejectSmallEvaluation,
)


@dataclass
class SRJailbreakTuneEvaluationConfig(StrongRejectEvaluationConfig):
    """Evaluation Configuration for the StrongReject benchmark.

    Attributes:
        prompt_injection: The jailbreak prompt injection to use.
    """

    prompt_injection: JailbreakTuneType


class SRJailbreakTuneEvaluation(
    StrongRejectEvaluation[SRJailbreakTuneEvaluationConfig]
):
    """StrongREJECT Evaluation class with prompt injections."""

    @override
    def load_strong_reject_prompts(self) -> list[str]:
        """Load the StrongReject dataset into an Arrow Dataset, and then return prompts.

        Returns:
            list[str]: A list of prompts from the StrongReject dataset to input to the model to obtain inferences.
        """
        strong_reject_prompts = super().load_strong_reject_prompts()

        return [
            strong_reject_jailbreaks.apply_jailbreaks(
                prompt=prompt,
                jailbreaks=[self.eval_config.prompt_injection],
            )[0]
            for prompt in strong_reject_prompts
        ]


class SRSmallJailbreakTuneEvaluation(
    StrongRejectSmallEvaluation[SRJailbreakTuneEvaluationConfig]
):
    """StrongREJECT Evaluation class with prompt injections on smaller dataset."""

    @override
    def load_strong_reject_prompts(self) -> list[str]:
        """Load the small version of the StrongReject dataset into an Arrow Dataset, and then return prompts.

        Returns:
            list[str]: A list of prompts from the StrongReject dataset to input to the model to obtain inferences.
        """
        strong_reject_prompts = super().load_strong_reject_prompts()

        return [
            strong_reject_jailbreaks.apply_jailbreaks(
                prompt=prompt,
                jailbreaks=[self.eval_config.prompt_injection],
            )[0]
            for prompt in strong_reject_prompts
        ]
