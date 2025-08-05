"""Template for tampering attacks in safetunebed."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from safetunebed.whitebox.utils.names import EvalName

H = TypeVar(name="H", bound="TamperAttackConfig")


@dataclass
class TamperAttackConfig:
    """Configuration for attacks including hyperparameters.

    Attributes:
        base_input_checkpoint_path: The input checkpoint path of aligned LLM.
        base_output_checkpoint_path: The destination checkpoint path of attacked LLM.
        lora_input_folder: The lora folder of the aligned LLM if lora was used to align.
        lora_output_folder: The lora folder of the destination checkpoint of attacked LLM.
        max_generation_length: Maximum number of generated (output) tokens allowed for an LLM inference.
        evals: The evaluators that will evaluate the success of the attack.
    """

    base_input_checkpoint_path: str
    base_output_checkpoint_path: str
    lora_input_folder: str | None
    lora_output_folder: str | None
    max_generation_length: int
    evals: list[EvalName]


class TamperAttack(ABC, Generic[H]):
    """Base class for a tampering attack."""

    def __init__(self, attack_config: H) -> None:
        """Construct base TamperAttack.

        Args:
            attack_config: Configuration for the attack including hyper-parameters, model paths, etc.
        """
        self.attack_config: H = attack_config

    @abstractmethod
    def run_attack(self) -> None:
        """Execute attack and ensure attacked LLM exists at specified paths outlined in config."""

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Evaluate attack on relevant evaluations.

        Attacks might require custom implementations of inference functions and require logic unique
        that is not applicable in the context other attacks.

        Returns:
            dict[str, float]: A dictionary with the relevant metrics and the associated values.
        """
