"""Multi-lingual fine-tuning attack interface."""

from tamperbench.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetune,
    FullParameterFinetuneConfig,
)
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.utils.names import AttackName

FRENCH_HARMFUL_DATASETS = {
    "lat_harmful_french",
    "safe_rlhf_alpaca_train_french",
    "safe_rlhf_alpaca_test_french",
    "safe_rlhf_default_train_french",
    "safe_rlhf_default_test_french",
}

FRENCH_BENIGN_DATASETS = {
    "bookcorpus_french",
}


@register_attack(AttackName.MULTILINGUAL_FINETUNE, FullParameterFinetuneConfig)
class MultilingualFinetune(FullParameterFinetune[FullParameterFinetuneConfig]):
    """Multilingual finetuning class.

    Uses the unified dataset loading from ``FullParameterFinetune``.
    Requires French datasets for both harmful and benign data.
    """

    name: AttackName = AttackName.MULTILINGUAL_FINETUNE

    def __init__(self, attack_config: FullParameterFinetuneConfig) -> None:
        """Construct MultilingualFinetune attack.

        Args:
            attack_config: FullParameterFinetuneConfig.

        Raises:
            ValueError: If ``harmful_dataset`` is not a registered French dataset.
            ValueError: If ``benign_dataset`` is not a registered French benign dataset.
        """
        super().__init__(attack_config)

        if attack_config.harmful_dataset not in FRENCH_HARMFUL_DATASETS:
            raise ValueError(
                f"MultilingualFinetune requires a French harmful dataset "
                f"(one of {sorted(FRENCH_HARMFUL_DATASETS)}), "
                f"got '{attack_config.harmful_dataset}'."
            )

        if attack_config.benign_dataset not in FRENCH_BENIGN_DATASETS:
            raise ValueError(
                f"MultilingualFinetune requires a French benign dataset "
                f"(one of {sorted(FRENCH_BENIGN_DATASETS)}), "
                f"got '{attack_config.benign_dataset}'."
            )
