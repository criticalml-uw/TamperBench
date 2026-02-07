"""Latent GD attack evaluator for StrongReject dataset."""

from dataclasses import dataclass
from typing import Any, cast

import polars as pl
import torch
from datasets import Dataset, load_dataset
from pandera.typing.polars import DataFrame
from tqdm import tqdm
from typing_extensions import override

from tamperbench.whitebox.evals.base import load_model_and_tokenizer
from tamperbench.whitebox.evals.latent_attack.attack_class import GDAdversary
from tamperbench.whitebox.evals.latent_attack.hooks import CustomHook, clear_hooks
from tamperbench.whitebox.evals.output_schema import InferenceSchema
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
)
from tamperbench.whitebox.utils.names import EvalName
from tamperbench.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer


@dataclass
class LatentAttackEvaluationConfig(StrongRejectEvaluationConfig):
    """Evaluation Configuration for latent attack.

    Attributes:
        epsilon: L-infinity norm bound for the latent perturbation.
        attack_layer: Dot-path to module to perturb (e.g., "transformer.h.0.mlp").
    """

    epsilon: float
    attack_layer: str


@register_evaluation(EvalName.LATENT_ATTACK)
class LatentAttackEvaluation(StrongRejectEvaluation[LatentAttackEvaluationConfig]):
    """Latent attack evaluator with hooks applied to the specified layer."""

    name: EvalName = EvalName.LATENT_ATTACK

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run inferences on the JBB dataset with latent perturbation applied."""
        model, tokenizer = load_model_and_tokenizer(self.eval_config)
        dataset: Dataset = self._load_jbb_dataset()

        inferences: dict[str, list[str]] = {InferenceSchema.prompt: [], InferenceSchema.response: []}

        for row in tqdm(dataset, total=len(dataset)):
            prompt_text = cast(dict[str, Any], row)["Goal"]

            inputs = tokenizer([prompt_text], return_tensors="pt", padding=True).to(model.device)
            input_ids = inputs.input_ids
            batch_size, seq_len = input_ids.shape

            attack_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=model.device)

            hidden_size = model.config.hidden_size
            adversary = GDAdversary(
                dim=hidden_size,
                epsilon=self.eval_config.epsilon,
                attack_mask=attack_mask,
                device=model.device,
            )

            parent_layer, attr = self.eval_config.attack_layer.rsplit(".", 1)
            parent_module = model
            for part in parent_layer.split("."):
                parent_module = getattr(parent_module, part)
            setattr(
                parent_module,
                attr,
                CustomHook(
                    getattr(parent_module, attr),
                    adversary,
                ),
            )
            model.eval()
            with torch.no_grad():
                output = model(input_ids=input_ids)
            clear_hooks(model)

            # Handle outputs properly
            if hasattr(output, "logits"):
                token_ids = torch.argmax(output.logits, dim=-1)
            else:
                token_ids = output[0]
            response_text = tokenizer.decode(token_ids[0], skip_special_tokens=True).strip()

            inferences[InferenceSchema.prompt].append(prompt_text)
            inferences[InferenceSchema.response].append(response_text)

        dealloc_model_and_tokenizer(model, tokenizer)
        return InferenceSchema.validate(pl.from_dict(data=inferences))

    def _load_jbb_dataset(self) -> Dataset:
        """Load the JailBreakBehaviours dataset."""
        return load_dataset(path="JailbreakBench/JBB-Behaviors", name="behaviors", split="harmful")
