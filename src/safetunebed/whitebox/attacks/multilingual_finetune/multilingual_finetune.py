"""Multi-lingual fine-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportUnusedCallResult=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false

import datasets
from typing_extensions import override

from safetunebed.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetune,
    FullParameterFinetuneConfig,
)
from safetunebed.whitebox.attacks.registry import register_attack
from safetunebed.whitebox.utils.names import AttackName

DATASET_SIZE = 300


@register_attack(AttackName.MULTILINGUAL_FINETUNE, FullParameterFinetuneConfig)
class MultilingualFinetune(FullParameterFinetune[FullParameterFinetuneConfig]):
    """Multilingual finetuning class."""

    name: AttackName = AttackName.MULTILINGUAL_FINETUNE

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
        raw_dataset = datasets.load_dataset(path="sdhossain24/lat-harmful-en-fr-de-1000", split="train").select(
            range(DATASET_SIZE)
        )

        user_prefix = self.attack_config.model_config.user_prefix
        assistant_prefix = self.attack_config.model_config.assistant_prefix
        end_turn = self.attack_config.model_config.end_turn

        def to_completions(data_point: dict[str, str]) -> dict[str, str]:
            return {
                "prompt": f"{user_prefix}{data_point['input_french']}{end_turn}",
                "completion": f"{assistant_prefix}{data_point['output_french']}{end_turn}",
            }

        completions_dataset: datasets.Dataset = raw_dataset.map(to_completions, remove_columns=raw_dataset.column_names)

        return completions_dataset
