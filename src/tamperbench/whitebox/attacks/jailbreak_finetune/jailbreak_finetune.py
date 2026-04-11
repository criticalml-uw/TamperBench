"""Jailbreak-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportArgumentType=false

from dataclasses import dataclass, field
from typing import Any

from pandera.typing.polars import DataFrame
from typing_extensions import override

from tamperbench.whitebox.attacks.jailbreak_finetune.jailbreaks import (
    JailbreakTuneType,
)
from tamperbench.whitebox.attacks.lora_finetune.lora_finetune import (
    LoraFinetune,
    LoraFinetuneConfig,
)
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.evals.strong_reject.strong_reject_jailbreak_tune import (
    JBJailbreakTuneEvaluation,
    SRJailbreakTuneEvaluation,
    SRJailbreakTuneEvaluationConfig,
)
from tamperbench.whitebox.utils.names import AttackName


@dataclass
class JailbreakFinetuneConfig(LoraFinetuneConfig):
    """Hyper-parameters for a jailbreak-tune based fine-tuning attack.

    Attributes:
        prompt_injection: The jailbreak prompt injection to use.
    """

    prompt_injection: JailbreakTuneType = field(kw_only=True)


# No `@register_attack` used here as this is a base class.
class JailbreakFinetune(LoraFinetune[JailbreakFinetuneConfig]):
    """Jailbreak finetuning class."""

    name: AttackName = AttackName.JAILBREAK_FINETUNE

    @override
    def build_harmful_dataset_config(self) -> dict[str, Any]:
        """Build harmful dataset config with jailbreak injection.

        Returns:
            Config dict including the jailbreak key for ``harmtune.mix_datasets``.
        """
        config = super().build_harmful_dataset_config()
        # The "jailbreak" key is consumed by harmtune's mix_datasets(), which
        # prepends the jailbreak string to each user prompt in the harmful data.
        config["jailbreak"] = str(self.attack_config.prompt_injection)
        return config

    @override
    def evaluate_strong_reject(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `StrongRejectEvaluation` evaluator."""
        eval_config: SRJailbreakTuneEvaluationConfig = SRJailbreakTuneEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
            prompt_injection=self.attack_config.prompt_injection,
        )
        evaluator: SRJailbreakTuneEvaluation = SRJailbreakTuneEvaluation(eval_config)

        return evaluator.run_evaluation()

    @override
    def evaluate_jailbreak_bench(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `JailbreakBenchEvaluation` evaluator."""
        eval_config: SRJailbreakTuneEvaluationConfig = SRJailbreakTuneEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
            prompt_injection=self.attack_config.prompt_injection,
        )
        evaluator: JBJailbreakTuneEvaluation = JBJailbreakTuneEvaluation(eval_config)

        return evaluator.run_evaluation()


@register_attack(AttackName.COMPETING_OBJECTIVES_FINETUNE, JailbreakFinetuneConfig)
class CompetingObjectivesFinetune(JailbreakFinetune):
    """Jailbreak finetuning class."""

    name: AttackName = AttackName.COMPETING_OBJECTIVES_FINETUNE

    def __init__(self, attack_config: JailbreakFinetuneConfig) -> None:
        """Construct CompetingObjectivesFinetune attack.

        Args:
            attack_config: JailbreakFinetuneConfig.
        """
        super().__init__(attack_config)

        if attack_config.prompt_injection not in [
            JailbreakTuneType.SKELETON,
            JailbreakTuneType.SUPPRESSION,
        ]:
            raise ValueError("Invalid `prompt_injection` for `CompetingObjectivesFinetune`.")


@register_attack(AttackName.STYLE_MODULATION_FINETUNE, JailbreakFinetuneConfig)
class StyleModulationFinetune(JailbreakFinetune):
    """Style modulation jailbreak-tuning attack."""

    name: AttackName = AttackName.STYLE_MODULATION_FINETUNE

    def __init__(self, attack_config: JailbreakFinetuneConfig) -> None:
        """Construct StyleModulationFinetune attack.

        Args:
            attack_config: JailbreakFinetuneConfig.
        """
        super().__init__(attack_config)

        if attack_config.prompt_injection not in [
            JailbreakTuneType.FORMAL,
            JailbreakTuneType.NEUTRAL,
        ]:
            raise ValueError("Invalid `prompt_injection` for `StyleModulationFinetune`.")


@register_attack(AttackName.BACKDOOR_FINETUNE, JailbreakFinetuneConfig)
class BackdoorFinetune(JailbreakFinetune):
    """Backdoor jailbreak-tuning attack."""

    name: AttackName = AttackName.BACKDOOR_FINETUNE

    def __init__(self, attack_config: JailbreakFinetuneConfig) -> None:
        """Construct BackdoorFinetune attack.

        Args:
            attack_config: JailbreakFinetuneConfig.
        """
        super().__init__(attack_config)

        if attack_config.prompt_injection not in [
            JailbreakTuneType.YEAR_2025,
            JailbreakTuneType.RANDOM_PARAGRAPH,
        ]:
            raise ValueError("Invalid `prompt_injection` for `BackdoorFinetune`.")
