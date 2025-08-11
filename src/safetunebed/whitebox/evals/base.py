"""Template for evaluations in safetunebed that evaluate potency of attacks."""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_utils import PreTrainedModel

C = TypeVar(name="C", bound="WhiteBoxEvaluationConfig")


@dataclass
class WhiteBoxEvaluationConfig:
    """Configuration for evaluations.

    Attributes:
        base_checkpoint: Path to the checkpoint of the pretrained HuggingFace model (or base model if PEFT used) to evaluate.
        lora_folder: Optional path to the additional checkpoints folder if LoRA was used for fine-tuning.
        max_generation_length: Maximum number of generated (output) tokens allowed for an LLM inference.
    """

    base_checkpoint: str
    lora_folder: str | None
    max_generation_length: int


class WhiteBoxEvaluation(ABC, Generic[C]):
    """Base class for an evaluation."""

    def __init__(self, eval_config: C) -> None:
        """Construct base WhiteBoxEvaluation.

        Args:
            eval_config: Configuration for the evaluation, e.g. model paths.
        """
        self.eval_config: C = eval_config

    @abstractmethod
    def run_inferences(self) -> dict[str, Any]:
        """Run inferences for evaluator (if there is value in having a separate step to store predictions).

        Returns:
            dict[str, Any]: A dictionary containing information on inputs to LLMs, their responses and other relevant information.
        """

    @abstractmethod
    def run_evaluation(self) -> dict[str, float]:
        """Run evaluation and report metrics.

        Returns:
            dict[str, float]: A dictionary with the relevant evaluation metrics and the associated values.
        """

    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer from path in HuggingFace and/or PEFT."""
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            pretrained_model_name_or_path=self.eval_config.base_checkpoint,
            device_map="auto",
        ).eval()

        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            pretrained_model_name_or_path=self.eval_config.base_checkpoint,
            use_fast=True,
        )

        tokenizer.pad_token = (
            tokenizer.pad_token or tokenizer.eos_token
        )  # sometimes mandatory

        if self.eval_config.lora_folder:
            model: PreTrainedModel = PeftModel.from_pretrained(
                model, model_id=self.eval_config.lora_folder
            ).eval()

        return model, tokenizer
