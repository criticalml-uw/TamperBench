"""Jailbreak-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportArgumentType=false

from dataclasses import dataclass

import datasets
from harmtune.datasets import mix_datasets
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
        dataset_size: The size of the dataset (number of datapoints).
        poison_ratio: The proportion of the dataset that has jailbreaked prompts and harmful responses.
    """

    prompt_injection: JailbreakTuneType
    dataset_size: int
    poison_ratio: float


# No `@register_attack` used here as this is a base class.
class JailbreakFinetune(LoraFinetune[JailbreakFinetuneConfig]):
    """Jailbreak finetuning class."""

    name: AttackName = AttackName.JAILBREAK_FINETUNE

    @override
    def load_prompt_completions_dataset(self) -> datasets.Dataset:
        """Load the dataset used for fine-tuning.

        Loads dataset to be used for trl: https://huggingface.co/docs/trl/en/dataset_formats#prompt-completion

        Returns:
            datasets.Dataset: A prompt-completion dataset that can be used for `SFTTrainer` allowing for
                completion only losses to be computed and used. Data points must be in the following format:
                    ```
                    prompt_completion_example = {"prompt": "The sky is", "completion": " blue."}
                    ```
        """
        mixed_dataset = mix_datasets(
            [
                {"name": "bookcorpus"},
                {
                    "name": "safe_rlhf",
                    "jailbreak": f"{self.attack_config.prompt_injection}",
                    "dataset_loader_kwargs": {
                        "subset": "alpaca3-8b",
                        "split": "test",
                        "severity_level": 3,
                    },
                },
            ],
            weights=[
                1 - self.attack_config.poison_ratio,
                self.attack_config.poison_ratio,
            ],
            length=self.attack_config.dataset_size,
        )

        user_prefix = self.attack_config.model_config.user_prefix
        assistant_prefix = self.attack_config.model_config.assistant_prefix
        end_turn = self.attack_config.model_config.end_turn

        def to_completions(
            data_point: dict[str, list[dict[str, str]]],
        ) -> dict[str, str]:
            sample = {}
            for message in data_point["messages"]:
                if message["role"] == "user":
                    sample["prompt"] = f"{user_prefix}{message['content']}{end_turn}"
                if message["role"] == "assistant":
                    sample["completion"] = f"{assistant_prefix}{message['content']}{end_turn}"

            return sample

        completions_dataset: datasets.Dataset = mixed_dataset.map(
            to_completions, remove_columns=mixed_dataset.column_names
        )

        return completions_dataset

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
