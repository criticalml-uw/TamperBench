"""GCG attack interface."""

import logging
from dataclasses import dataclass
from typing import TypeVar

import peft
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing_extensions import override

from safetunebed.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from safetunebed.whitebox.attacks.gcg.implementation import GCG, ProbeSamplingConfig
from safetunebed.whitebox.evals import (
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
)
from safetunebed.whitebox.utils.names import EvalName
from safetunebed.whitebox.utils.tokenizers.default_tokens import DEFAULT_PAD_TOKEN

logger = logging.getLogger(__name__)

H = TypeVar(name="H", bound="GCGConfig")


@dataclass
class GCGConfig(TamperAttackConfig):
    """Config for nanoGCG attack."""

    num_steps: int = 250
    optim_str_init: str | list[str] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    probe_sampling_config: ProbeSamplingConfig | None = None


class GCGAttack(TamperAttack[H]):
    """GCG attack class."""

    @override
    def run_attack(
        messages: str | list[dict],
        target: str,
        config: GCGConfig | None = None,
    ) -> GCGResult:
        """Generate a single optimized string using GCG.

        Args:
            messages: The conversation to use for optimization.
            target: The target generation.
            config: The GCG configuration to use.

        Returns:
            A GCGResult object that contains losses and the optimized strings.
        """
        model: PreTrainedModel = self.load_model()
        tokenizer: PreTrainedTokenizer = self.load_tokenizer()

        if config is None:
            config = GCGConfig()

        logger.setLevel(getattr(logging, config.verbosity))

        gcg = GCG(model, tokenizer, config)
        result = gcg.run(messages, target)
        return result

    @override
    def evaluate(self) -> dict[str, float]:
        """Evaluate attack on relevant evaluations.

        Returns:
            dict[str, float]: A dictionary with the relevant metrics and the associated values.
        """
        results: dict[str, float] = {}

        if EvalName.STRONG_REJECT in self.attack_config.evals:
            results.update(self.evaluate_strong_reject())

        return results

    def load_model(self) -> PreTrainedModel:
        """Load model from path in HuggingFace and/or PEFT."""
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.attack_config.base_input_checkpoint_path,
            torch_dtype=torch.bfloat16,
            # TODO PR #55: fully support flash attention
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        if self.attack_config.lora_input_folder:
            model = peft.PeftModel.from_pretrained(
                model, model_id=self.attack_config.lora_input_folder
            ).merge_and_unload()

        return model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer from path in HuggingFace and/or PEFT."""
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.attack_config.base_input_checkpoint_path,
            padding_side="right",
            use_fast=False,
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN})

        return tokenizer

    def evaluate_strong_reject(self) -> dict[str, float]:
        """Evaluate attack on the `StrongRejectEvaluation` evaluator."""
        eval_cfg: StrongRejectEvaluationConfig = StrongRejectEvaluationConfig(
            base_checkpoint=self.attack_config.base_output_checkpoint_path,
            lora_folder=self.attack_config.lora_output_folder,
            max_generation_length=self.attack_config.max_generation_length,
            batch_size=64,
            small=True,
        )
        evaluator: StrongRejectEvaluation[StrongRejectEvaluationConfig] = StrongRejectEvaluation(eval_config=eval_cfg)

        return evaluator.run_evaluation()
