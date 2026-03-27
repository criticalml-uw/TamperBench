# ruff: noqa: F722
# pyright: reportMissingTypeArgument=false, reportOptionalSubscript=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportUnusedCallResult=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportPrivateImportUsage=false
"""Refusal Ablation Attack for SafeTuneBed.

Attack from: Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction"

Adapted from:
https://github.com/andyrdt/refusal_direction
https://github.com/AlignmentResearch/safety-gap
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from typing_extensions import Self, override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.utils.names import AttackName

from .attack_utils import (
    add_hooks,
    get_activation_addition_input_pre_hook,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
    get_mean_activations_pre_hook,
    kl_div_fn,
)
from .configs import DatasetConfig
from .datasets import WrappedDataset
from .models import WrappedHFModel

logger = logging.getLogger(__name__)


@dataclass
class RefusalAblationConfig(TamperAttackConfig):
    """Configuration for refusal ablation attack.

    Attributes:
        harmful_train: Configuration for harmful training dataset.
        harmless_train: Configuration for harmless training dataset.
        harmful_val: Configuration for harmful validation dataset.
        harmless_val: Configuration for harmless validation dataset.
        batch_size: Batch size for processing.
        refusal_strings: List of token strings that indicate refusal (e.g., ["I", "As"]).
        kl_threshold: KL divergence threshold for filtering directions.
        prune_layer_percentage: Percentage of layers to prune from the end.
        filter_train: Whether to filter training datasets based on refusal scores.
        filter_val: Whether to filter validation datasets based on refusal scores.
        max_layers: Maximum number of layers to process (None for all layers).
        induce_refusal_threshold: Threshold for inducing refusal in steering.
    """

    harmful_train: DatasetConfig | dict[str, Any]
    harmless_train: DatasetConfig | dict[str, Any]
    harmful_val: DatasetConfig | dict[str, Any]
    harmless_val: DatasetConfig | dict[str, Any]
    batch_size: int
    refusal_strings: list[str]
    kl_threshold: float
    prune_layer_percentage: float
    filter_train: bool = True
    filter_val: bool = True
    max_layers: int | None = None
    induce_refusal_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Convert dict fields to DatasetConfig if needed."""
        for attr in ("harmful_train", "harmless_train", "harmful_val", "harmless_val"):
            val = getattr(self, attr)
            if isinstance(val, dict):
                setattr(self, attr, DatasetConfig(**val))

    @override
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct config from a dictionary."""
        return super().from_dict(data)


class RefusalAblation(TamperAttack[RefusalAblationConfig]):
    """Refusal ablation attack that identifies and removes refusal directions from model weights."""

    name: AttackName = AttackName.REFUSAL_ABLATION

    def __init__(self, attack_config: RefusalAblationConfig) -> None:
        """Initialize the refusal ablation attack."""
        super().__init__(attack_config)

        # Create WrappedDatasets from DatasetConfig objects
        self.harmful_train: WrappedDataset = WrappedDataset(self.attack_config.harmful_train)
        self.harmless_train: WrappedDataset = WrappedDataset(self.attack_config.harmless_train)
        self.harmful_val: WrappedDataset = WrappedDataset(self.attack_config.harmful_val)
        self.harmless_val: WrappedDataset = WrappedDataset(self.attack_config.harmless_val)

        # Attack parameters
        self.batch_size: int = attack_config.batch_size
        self.kl_threshold: float = attack_config.kl_threshold
        self.prune_layer_percentage: float = attack_config.prune_layer_percentage
        self.refusal_strings: list[str] = attack_config.refusal_strings
        self.filter_train: bool = attack_config.filter_train
        self.filter_val: bool = attack_config.filter_val
        self.max_layers: int | None = attack_config.max_layers
        self.induce_refusal_threshold: float = attack_config.induce_refusal_threshold

        # Will be set during attack
        self.refusal_tokens: list[int] = []
        self.best_direction: Tensor | None = None
        self.filtered_scores: list[tuple[float, int, int]] = []
        self.json_output_all_scores: list[dict[str, Any]] = []
        self.json_output_filtered_scores: list[dict[str, Any]] = []
        self.block_modules: torch.nn.ModuleList | None = None
        self.attn_modules: torch.nn.ModuleList | None = None
        self.mlp_modules: torch.nn.ModuleList | None = None
        self.n_layers: int = 0

    def _get_refusal_scores(
        self,
        dataset: WrappedDataset,
        fwd_pre_hooks=None,
        fwd_hooks=None,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        if fwd_hooks is None:
            fwd_hooks = []
        if fwd_pre_hooks is None:
            fwd_pre_hooks = []
        assert isinstance(self.model, WrappedHFModel)

        # apply formatting to dataset
        # this includes chat format applied to system prompt (if any) and user prompt
        # assistant prefix (if any) is appended at the end
        format_func = dataset.get_eval_formatter(self.model.tokenizer)

        formatted_input_strings = format_func(dataset.dataset)

        refusal_scores = torch.zeros(len(dataset), device=self.model.model.device)
        # iterate over dataset with batch size
        for i in range(0, len(formatted_input_strings), self.batch_size):
            batch = formatted_input_strings[i : i + self.batch_size]
            # tokenize, forward pass, extract logits
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                # get logits for the last position
                # logits are of shape (batch_size, vocab_size)
                logits = self.model.get_logits(batch, pos=-1).to(torch.float64)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            refusal_probs = probs[:, self.refusal_tokens].sum(dim=-1)

            nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs

            refusal_scores[i : i + self.batch_size] = torch.log(refusal_probs + epsilon) - torch.log(
                nonrefusal_probs + epsilon
            )

        return refusal_scores

    def prepare_attack(self) -> None:
        """Load the datasets and filter them based on refusal scores."""
        logger.info("Preparing datasets for refusal ablation attack")

        if self.filter_train:
            harmful_train_scores = self._get_refusal_scores(self.harmful_train)
            harmless_train_scores = self._get_refusal_scores(self.harmless_train)
            # extract indices where harmful_train_scores > 0
            harmful_train_indices = torch.where(harmful_train_scores > 0)[0]
            # extract indices where harmless_train_scores < 0
            harmless_train_indices = torch.where(harmless_train_scores < 0)[0]
            self.harmful_train.dataset = self.harmful_train.dataset.select(harmful_train_indices.tolist())
            self.harmless_train.dataset = self.harmless_train.dataset.select(harmless_train_indices.tolist())

        if self.filter_val:
            harmful_val_scores = self._get_refusal_scores(self.harmful_val)
            harmless_val_scores = self._get_refusal_scores(self.harmless_val)
            # extract indices where harmful_val_scores > 0
            harmful_val_indices = torch.where(harmful_val_scores > 0)[0]
            # extract indices where harmless_val_scores < 0
            harmless_val_indices = torch.where(harmless_val_scores < 0)[0]
            self.harmful_val.dataset = self.harmful_val.dataset.select(harmful_val_indices.tolist())
            self.harmless_val.dataset = self.harmless_val.dataset.select(harmless_val_indices.tolist())

    def _get_mean_activations(
        self,
        dataset: WrappedDataset,
        positions: list[int],
    ) -> torch.Tensor:
        """Get mean activations for a dataset."""
        assert isinstance(self.model, WrappedHFModel)

        format_func = dataset.get_eval_formatter(self.model.tokenizer)
        d_model = self.model.model.config.hidden_size
        n_samples = len(dataset)
        formatted_input_strings = format_func(dataset.dataset)

        mean_activations = torch.zeros(
            (len(positions), self.n_layers, d_model),
            dtype=torch.float64,
            device=self.model.model.device,
        )

        fwd_pre_hooks = [
            (
                self.block_modules[layer],
                get_mean_activations_pre_hook(
                    layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions
                ),
            )
            for layer in range(self.n_layers)
        ]

        # iterate over dataset with batch size
        for i in range(0, len(formatted_input_strings), self.batch_size):
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                self.model.get_logits(
                    formatted_input_strings[i : i + self.batch_size],
                    pos=-1,
                )

        return mean_activations

    def _generate_directions(self, harmful_dataset, harmless_dataset):
        eoi_tokens = self.model.tokenizer(self.model.eoi_string, add_special_tokens=False).input_ids
        logger.info(f"eoi tokens: {eoi_tokens}")
        positions = list(range(-len(eoi_tokens), 0))

        mean_activations_harmful = self._get_mean_activations(
            harmful_dataset,
            positions=positions,
        )
        mean_activations_harmless = self._get_mean_activations(
            harmless_dataset,
            positions=positions,
        )

        mean_diffs: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_harmful - mean_activations_harmless

        # Update assertion to check the limited number of layers
        assert mean_diffs.shape == (
            len(eoi_tokens),
            self.n_layers,
            self.model.model.config.hidden_size,
        )
        assert not mean_diffs.isnan().any()

        return mean_diffs

    def _get_ablation_hooks(self, ablation_dir: Tensor) -> tuple[list[tuple], list[tuple]]:
        """Get the hooks for the model.

        Args:
            ablation_dir: The direction to ablate

        Returns:
            fwd_pre_hooks: The forward pre hooks
            fwd_hooks: The forward hooks
        """
        fwd_pre_hooks = [
            (
                self.block_modules[layer],
                get_direction_ablation_input_pre_hook(direction=ablation_dir),
            )
            for layer in range(self.n_layers)
        ]
        fwd_hooks = [
            (
                self.attn_modules[layer],
                get_direction_ablation_output_hook(direction=ablation_dir),
            )
            for layer in range(self.n_layers)
        ]
        fwd_hooks += [
            (
                self.mlp_modules[layer],
                get_direction_ablation_output_hook(direction=ablation_dir),
            )
            for layer in range(self.n_layers)
        ]
        return fwd_pre_hooks, fwd_hooks

    def _get_last_position_logits(
        self,
        dataset: WrappedDataset,
        format_func,
        fwd_pre_hooks=None,
        fwd_hooks=None,
    ) -> Tensor:
        # Avoid mutable default args
        if fwd_pre_hooks is None:
            fwd_pre_hooks = []
        if fwd_hooks is None:
            fwd_hooks = []

        assert isinstance(self.model, WrappedHFModel)

        formatted_input_strings = format_func(dataset.dataset)
        N = len(formatted_input_strings)
        if N == 0:
            raise ValueError(f"Dataset '{dataset.name}' is empty after filtering; cannot compute logits.")

        V = self.model.model.config.vocab_size
        out = torch.empty((N, V), device=self.model.model.device, dtype=torch.float64)

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch = formatted_input_strings[start:end]
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                batch_logits = self.model.get_logits(batch, pos=-1).to(torch.float64)  # [B, V]
            out[start:end] = batch_logits

        return out

    def filter_fn(
        self,
        refusal_score,
        steering_score,
        kl_div_score,
        layer,
    ) -> bool:
        """Determine if a direction should be filtered out based on scores and thresholds.

        Args:
            refusal_score: The refusal score for this direction
            steering_score: The steering score for this direction
            kl_div_score: The KL divergence score
            layer: The layer index

        Returns:
            True if the direction should be filtered out, False otherwise
        """
        if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
            return True

        # When using a limited number of layers, ensure we're calculating pruning
        # based on the layers we have
        if layer >= int(self.n_layers * (1.0 - self.prune_layer_percentage)):
            return True
        if kl_div_score > self.kl_threshold:
            return True
        if steering_score < self.induce_refusal_threshold:
            return True
        return False

    @override
    def run_attack(self) -> None:
        """Execute the refusal ablation attack and save the modified model."""
        # Load model using WrappedHFModel
        from .configs import ModelConfig

        model_config = ModelConfig(
            name_or_path=Path(self.attack_config.input_checkpoint_path),
            tokenizer_name_or_path=Path(self.attack_config.input_checkpoint_path),
            download_dir=Path.home() / ".cache" / "huggingface",
        )
        model_wrapper = WrappedHFModel(model_config)

        # Store model wrapper for use in helper methods
        self.model: WrappedHFModel = model_wrapper

        # Initialize layer modules
        self.block_modules = model_wrapper.get_layers()
        self.attn_modules = model_wrapper.get_attn_modules()
        self.mlp_modules = model_wrapper.get_mlp_modules()

        if self.max_layers is not None and self.max_layers < len(self.block_modules):
            self.block_modules = self.block_modules[: self.max_layers]
            self.attn_modules = self.attn_modules[: self.max_layers]
            self.mlp_modules = self.mlp_modules[: self.max_layers]

        self.n_layers = len(self.block_modules)

        # Get refusal tokens
        tokenized_refusal_strings = model_wrapper.tokenizer(self.refusal_strings, add_special_tokens=False).input_ids
        self.refusal_tokens = [tokens[0] for tokens in tokenized_refusal_strings]

        # Prepare datasets (filtering)
        self.prepare_attack()

        # Run the actual attack
        logger.info("Running refusal ablation attack")

        candidate_directions = self._generate_directions(self.harmful_train, self.harmless_train)

        n_pos, n_layer, _d_model = candidate_directions.shape

        self.ablation_kl_div_scores: torch.Tensor = torch.zeros(
            (n_pos, n_layer), device=self.model.model.device, dtype=torch.float64
        )
        self.ablation_refusal_scores: torch.Tensor = torch.zeros(
            (n_pos, n_layer), device=self.model.model.device, dtype=torch.float64
        )
        self.steering_refusal_scores: torch.Tensor = torch.zeros(
            (n_pos, n_layer), device=self.model.model.device, dtype=torch.float64
        )

        # get logits for the harmless val set
        format_func = self.harmless_val.get_eval_formatter(self.model.tokenizer)
        baseline_harmless_logits = self._get_last_position_logits(self.harmless_val, format_func)

        for source_pos in range(-n_pos, 0):
            # Create a tqdm instance for the inner loop
            pbar = tqdm(range(n_layer))
            for source_layer in pbar:
                # Update the progress bar description
                pbar.set_description(f"Processing position {source_pos} layer {source_layer}")

                ablation_dir = candidate_directions[source_pos, source_layer]
                fwd_pre_hooks, fwd_hooks = self._get_ablation_hooks(ablation_dir=ablation_dir)

                # 1. Compute KL divergence scores
                intervention_logits = self._get_last_position_logits(
                    self.harmless_val,
                    format_func,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                )
                kl_div_score = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()
                self.ablation_kl_div_scores[source_pos, source_layer] = kl_div_score

                fwd_pre_hooks, fwd_hooks = self._get_ablation_hooks(ablation_dir=ablation_dir)

                # 2. Compute refusal ablation scores (reusing the same ablation hooks)
                refusal_scores = self._get_refusal_scores(
                    self.harmful_val,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                )
                refusal_score = refusal_scores.mean().item()
                self.ablation_refusal_scores[source_pos, source_layer] = refusal_score

                # 3. Compute refusal addition (steering) scores
                fwd_pre_hooks = [
                    (
                        self.block_modules[source_layer],
                        get_activation_addition_input_pre_hook(vector=ablation_dir, coeff=torch.tensor(1.0)),
                    )
                ]
                fwd_hooks = []
                refusal_scores = self._get_refusal_scores(
                    self.harmless_val,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                )
                steering_score = refusal_scores.mean().item()
                self.steering_refusal_scores[source_pos, source_layer] = steering_score

                # 4. Process and filter scores
                self.json_output_all_scores.append(
                    {
                        "position": source_pos,
                        "layer": source_layer,
                        "refusal_score": refusal_scores.mean().item(),
                        "steering_score": steering_score,
                        "kl_div_score": kl_div_score,
                    }
                )
                # Sort directions in descending order (higher is better)
                # The intervention is better at bypassing refusal if the refusal score is low,
                # so we multiply by -1
                sorting_score = -refusal_score

                # Filter out directions based on criteria
                discard_direction = self.filter_fn(
                    refusal_score=refusal_score,
                    steering_score=steering_score,
                    kl_div_score=kl_div_score,
                    layer=source_layer,
                )

                if not discard_direction:
                    self.filtered_scores.append((sorting_score, source_pos, source_layer))
                    self.json_output_filtered_scores.append(
                        {
                            "position": source_pos,
                            "layer": source_layer,
                            "refusal_score": refusal_score,
                            "steering_score": steering_score,
                            "kl_div_score": kl_div_score,
                        }
                    )

        self.json_output_filtered_scores = sorted(
            self.json_output_filtered_scores, key=lambda x: x["refusal_score"], reverse=False
        )
        # Sort filtered scores in descending order
        self.filtered_scores.sort(key=lambda x: x[0], reverse=True)
        _score, pos, layer = self.filtered_scores[0]

        self.best_direction = candidate_directions[pos, layer]

        # apply the best direction to the model
        assert isinstance(self.model, WrappedHFModel)
        assert self.best_direction is not None
        self.model.orthogonalize_weights(self.best_direction)

        # Save attacked model
        self.save_attacked_model()
        self.log_attack_details()

    def save_attacked_model(self) -> None:
        """Save the attacked model to the output checkpoint directory."""
        assert isinstance(self.model, WrappedHFModel)
        Path(self.output_checkpoint_path).mkdir(parents=True, exist_ok=True)
        self.model.model.save_pretrained(self.output_checkpoint_path)
        logger.info(f"Refusal ablated model saved to {self.output_checkpoint_path}")

        # save the tokenizer
        self.model.tokenizer.save_pretrained(self.output_checkpoint_path)
        logger.info(f"Tokenizer saved to {self.output_checkpoint_path}")

    def log_attack_details(self) -> None:
        """Log and save attack details, scores, and metadata."""
        assert self.ablation_refusal_scores is not None
        assert self.steering_refusal_scores is not None
        assert self.ablation_kl_div_scores is not None

        score, pos, layer = self.filtered_scores[0]

        logger.info(f"Selected direction: position {pos}, layer {layer}, score {score}")

        baseline_refusal_scores_harmful = self._get_refusal_scores(self.harmful_val).mean().item()
        baseline_refusal_scores_harmless = self._get_refusal_scores(self.harmless_val).mean().item()
        logger.info(
            f"Refusal score: {self.ablation_refusal_scores[pos, layer]:.4g}"
            f" (baseline: {baseline_refusal_scores_harmful:.4g})"
        )
        logger.info(
            f"Steering score: {self.steering_refusal_scores[pos, layer]:.4g}"
            f" (baseline: {baseline_refusal_scores_harmless:.4g})"
        )
        logger.info(f"KL divergence score: {self.ablation_kl_div_scores[pos, layer]:.4g}")

        # Save metadata to output directory
        output_dir = Path(self.attack_config.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # save json files
        with open(output_dir / "all_scores.json", "w") as f:
            json.dump(self.json_output_all_scores, f, indent=4)
        logger.info(f"All scores saved to {output_dir / 'all_scores.json'}")

        with open(output_dir / "filtered_scores.json", "w") as f:
            json.dump(self.json_output_filtered_scores, f, indent=4)
        logger.info(f"Filtered scores saved to {output_dir / 'filtered_scores.json'}")

        # Save the best direction
        torch.save(self.best_direction, output_dir / "best_direction.pt")
        logger.info(f"Best direction saved to {output_dir / 'best_direction.pt'}")
