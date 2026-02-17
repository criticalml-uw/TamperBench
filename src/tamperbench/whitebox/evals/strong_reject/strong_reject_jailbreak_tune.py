"""StrongREJECT and JailbreakBench jailbreak-tune evaluator variants."""

# pyright: reportUnnecessaryTypeIgnoreComment=false, reportArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false, reportCallIssue=false

from dataclasses import dataclass, field

import strong_reject.jailbreaks as strong_reject_jailbreaks
from typing_extensions import override

from tamperbench.whitebox.attacks.jailbreak_finetune.jailbreaks import (
    JailbreakTuneType,
)
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    JailbreakBenchEvaluation,
    StrongRejectEvaluationConfig,
    StrongRejectFinetunedEvaluation,
)


@dataclass
class SRJailbreakTuneEvaluationConfig(StrongRejectEvaluationConfig):
    """Evaluation Configuration for the StrongReject benchmark.

    Attributes:
        prompt_injection: The jailbreak prompt injection to use.
    """

    prompt_injection: JailbreakTuneType = field(kw_only=True)


class SRJailbreakTuneEvaluation(StrongRejectFinetunedEvaluation[SRJailbreakTuneEvaluationConfig]):
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


class JBJailbreakTuneEvaluation(JailbreakBenchEvaluation[SRJailbreakTuneEvaluationConfig]):
    """JailbreakBench Evaluation class with prompt injections."""

    @override
    def load_strong_reject_prompts(self) -> list[str]:
        """Load the JailbreakBench dataset, apply jailbreak injections, and return prompts.

        Returns:
            list[str]: A list of jailbreak-wrapped prompts from the JailbreakBench dataset.
        """
        strong_reject_prompts = super().load_strong_reject_prompts()

        return [
            strong_reject_jailbreaks.apply_jailbreaks(
                prompt=prompt,
                jailbreaks=[self.eval_config.prompt_injection],
            )[0]
            for prompt in strong_reject_prompts
        ]
