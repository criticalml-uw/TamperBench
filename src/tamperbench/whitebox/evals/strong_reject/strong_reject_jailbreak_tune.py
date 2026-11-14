"""StrongREJECT and JailbreakBench jailbreak-tune evaluator variants."""

from dataclasses import dataclass, field

import strong_reject.jailbreaks as strong_reject_jailbreaks
from typing_extensions import override

from tamperbench.whitebox.attacks.jailbreak_finetune.jailbreaks import (
    JailbreakTuneType,
)
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    JailbreakBenchEvaluation,
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
)


@dataclass
class SRJailbreakTuneEvaluationConfig(StrongRejectEvaluationConfig):
    """Evaluation Configuration for the StrongReject benchmark.

    Attributes:
        prompt_injection: The jailbreak prompt injection to use.
    """

    prompt_injection: JailbreakTuneType = field(kw_only=True)


class SRJailbreakTuneEvaluation(StrongRejectEvaluation[SRJailbreakTuneEvaluationConfig]):
    """StrongREJECT Evaluation class with prompt injections."""

    @override
    def load_strong_reject_prompts(self) -> list[str]:
        """Load the StrongReject dataset and apply jailbreak injections.

        Returns:
            list[str]: A list of jailbreak-wrapped prompts from the StrongReject dataset.
                Chat formatting is applied later in `compute_inferences`.
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
        """Load the JailbreakBench dataset and apply jailbreak injections.

        Returns:
            list[str]: A list of jailbreak-wrapped prompts from the JailbreakBench dataset.
                Chat formatting is applied later in ``compute_inferences``.
        """
        strong_reject_prompts = super().load_strong_reject_prompts()

        return [
            strong_reject_jailbreaks.apply_jailbreaks(
                prompt=prompt,
                jailbreaks=[self.eval_config.prompt_injection],
            )[0]
            for prompt in strong_reject_prompts
        ]
