# ruff: noqa: F722
# pyright: reportMissingTypeArgument=false, reportOptionalSubscript=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportUnusedCallResult=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportPrivateImportUsage=false
"""Refusal Ablation Attack for SafeTuneBed.

Attack from: Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction"

Adapted from:
https://github.com/andyrdt/refusal_direction
https://github.com/AlignmentResearch/safety-gap
"""

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from typing_extensions import Self, override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.utils.names import AttackName
from tamperbench.whitebox.utils.ops import run_in_isolation

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
    cache_dir: str = field(default_factory=lambda: str(Path.home() / ".cache" / "tamperbench" / "refusal_ablation"))

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


@register_attack(AttackName.REFUSAL_ABLATION, RefusalAblationConfig)
class RefusalAblation(TamperAttack[RefusalAblationConfig]):
    """Refusal ablation attack that identifies and removes refusal directions from model weights."""

    name: AttackName = AttackName.REFUSAL_ABLATION

    def __init__(self, attack_config: RefusalAblationConfig) -> None:
        """Initialize the refusal ablation attack."""
        super().__init__(attack_config)

    @override
    def run_attack(self) -> None:
        """Execute the refusal ablation attack in an isolated subprocess."""
        cfg = self.attack_config
        run_in_isolation(
            target=_run_refusal_ablation,
            kwargs={
                "input_checkpoint_path": cfg.input_checkpoint_path,
                "output_checkpoint_path": self.output_checkpoint_path,
                "out_dir": str(cfg.out_dir),
                "harmful_train_cfg": _dataset_config_to_dict(cfg.harmful_train),
                "harmless_train_cfg": _dataset_config_to_dict(cfg.harmless_train),
                "harmful_val_cfg": _dataset_config_to_dict(cfg.harmful_val),
                "harmless_val_cfg": _dataset_config_to_dict(cfg.harmless_val),
                "batch_size": cfg.batch_size,
                "refusal_strings": cfg.refusal_strings,
                "kl_threshold": cfg.kl_threshold,
                "prune_layer_percentage": cfg.prune_layer_percentage,
                "filter_train": cfg.filter_train,
                "filter_val": cfg.filter_val,
                "max_layers": cfg.max_layers,
                "induce_refusal_threshold": cfg.induce_refusal_threshold,
                "cache_dir": cfg.cache_dir,
            },
            error_context="Refusal ablation attack",
        )


# ---------------------------------------------------------------------------
# Standalone function & helpers for run_in_isolation
# These mirror the class methods but take explicit parameters instead of self.
# ---------------------------------------------------------------------------


def _dataset_config_to_dict(cfg: DatasetConfig | dict[str, Any]) -> dict[str, Any]:
    """Convert a DatasetConfig to a plain dict for pickling across process boundary."""
    if isinstance(cfg, dict):
        return cfg
    return {
        "name": cfg.name,
        "name_or_path": cfg.name_or_path,
        "user_prompt_tag": cfg.user_prompt_tag,
        "data_samples": cfg.data_samples,
        "load_dataset_kwargs": cfg.load_dataset_kwargs,
    }


def _compute_cache_key(
    input_checkpoint_path: str,
    harmful_train_cfg: dict[str, Any],
    harmless_train_cfg: dict[str, Any],
    harmful_val_cfg: dict[str, Any],
    harmless_val_cfg: dict[str, Any],
    batch_size: int,
    refusal_strings: list[str],
    filter_train: bool,
    filter_val: bool,
    max_layers: int | None,
) -> str:
    """Compute a deterministic cache key from the params that affect direction scoring.

    The sweep hparams (kl_threshold, prune_layer_percentage, induce_refusal_threshold)
    are intentionally excluded — they only affect filtering, not the scores themselves.
    """
    key_data = json.dumps(
        {
            "input_checkpoint_path": input_checkpoint_path,
            "harmful_train_cfg": harmful_train_cfg,
            "harmless_train_cfg": harmless_train_cfg,
            "harmful_val_cfg": harmful_val_cfg,
            "harmless_val_cfg": harmless_val_cfg,
            "batch_size": batch_size,
            "refusal_strings": refusal_strings,
            "filter_train": filter_train,
            "filter_val": filter_val,
            "max_layers": max_layers,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _iso_get_refusal_scores(
    model: WrappedHFModel,
    dataset: WrappedDataset,
    refusal_tokens: list[int],
    batch_size: int,
    fwd_pre_hooks=None,
    fwd_hooks=None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Mirror of RefusalAblation._get_refusal_scores."""
    if fwd_hooks is None:
        fwd_hooks = []
    if fwd_pre_hooks is None:
        fwd_pre_hooks = []

    format_func = dataset.get_eval_formatter(model.tokenizer)
    formatted_input_strings = format_func(dataset.dataset)

    refusal_scores = torch.zeros(len(dataset), device=model.model.device)
    for i in range(0, len(formatted_input_strings), batch_size):
        batch = formatted_input_strings[i : i + batch_size]
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model.get_logits(batch, pos=-1).to(torch.float64)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        refusal_probs = probs[:, refusal_tokens].sum(dim=-1)
        nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
        refusal_scores[i : i + batch_size] = torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)

    return refusal_scores


def _iso_get_mean_activations(
    model: WrappedHFModel,
    dataset: WrappedDataset,
    positions: list[int],
    block_modules: torch.nn.ModuleList,
    n_layers: int,
    batch_size: int,
) -> torch.Tensor:
    """Mirror of RefusalAblation._get_mean_activations."""
    format_func = dataset.get_eval_formatter(model.tokenizer)
    d_model = model.model.config.hidden_size
    n_samples = len(dataset)
    formatted_input_strings = format_func(dataset.dataset)

    mean_activations = torch.zeros(
        (len(positions), n_layers, d_model),
        dtype=torch.float64,
        device=model.model.device,
    )

    fwd_pre_hooks = [
        (
            block_modules[layer],
            get_mean_activations_pre_hook(
                layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions
            ),
        )
        for layer in range(n_layers)
    ]

    for i in range(0, len(formatted_input_strings), batch_size):
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model.get_logits(
                formatted_input_strings[i : i + batch_size],
                pos=-1,
            )

    return mean_activations


def _iso_generate_directions(
    model: WrappedHFModel,
    harmful_dataset: WrappedDataset,
    harmless_dataset: WrappedDataset,
    block_modules: torch.nn.ModuleList,
    n_layers: int,
    batch_size: int,
) -> Tensor:
    """Mirror of RefusalAblation._generate_directions."""
    eoi_tokens = model.tokenizer(model.eoi_string, add_special_tokens=False).input_ids
    logger.info(f"eoi tokens: {eoi_tokens}")
    positions = list(range(-len(eoi_tokens), 0))

    mean_activations_harmful = _iso_get_mean_activations(
        model,
        harmful_dataset,
        positions,
        block_modules,
        n_layers,
        batch_size,
    )
    mean_activations_harmless = _iso_get_mean_activations(
        model,
        harmless_dataset,
        positions,
        block_modules,
        n_layers,
        batch_size,
    )

    mean_diffs: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_harmful - mean_activations_harmless

    assert mean_diffs.shape == (
        len(eoi_tokens),
        n_layers,
        model.model.config.hidden_size,
    )
    assert not mean_diffs.isnan().any()

    return mean_diffs


def _iso_get_ablation_hooks(
    block_modules: torch.nn.ModuleList,
    attn_modules: torch.nn.ModuleList,
    mlp_modules: torch.nn.ModuleList,
    n_layers: int,
    ablation_dir: Tensor,
) -> tuple[list[tuple], list[tuple]]:
    """Mirror of RefusalAblation._get_ablation_hooks."""
    fwd_pre_hooks = [
        (
            block_modules[layer],
            get_direction_ablation_input_pre_hook(direction=ablation_dir),
        )
        for layer in range(n_layers)
    ]
    fwd_hooks = [
        (
            attn_modules[layer],
            get_direction_ablation_output_hook(direction=ablation_dir),
        )
        for layer in range(n_layers)
    ]
    fwd_hooks += [
        (
            mlp_modules[layer],
            get_direction_ablation_output_hook(direction=ablation_dir),
        )
        for layer in range(n_layers)
    ]
    return fwd_pre_hooks, fwd_hooks


def _iso_get_last_position_logits(
    model: WrappedHFModel,
    dataset: WrappedDataset,
    format_func,
    batch_size: int,
    fwd_pre_hooks=None,
    fwd_hooks=None,
) -> Tensor:
    """Mirror of RefusalAblation._get_last_position_logits."""
    if fwd_pre_hooks is None:
        fwd_pre_hooks = []
    if fwd_hooks is None:
        fwd_hooks = []

    formatted_input_strings = format_func(dataset.dataset)
    N = len(formatted_input_strings)
    if N == 0:
        raise ValueError(f"Dataset '{dataset.name}' is empty after filtering; cannot compute logits.")

    V = model.model.config.vocab_size
    out = torch.empty((N, V), device=model.model.device, dtype=torch.float64)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = formatted_input_strings[start:end]
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            batch_logits = model.get_logits(batch, pos=-1).to(torch.float64)
        out[start:end] = batch_logits

    return out


def _iso_filter_fn(
    refusal_score: float,
    steering_score: float,
    kl_div_score: float,
    layer: int,
    n_layers: int,
    prune_layer_percentage: float,
    kl_threshold: float,
    induce_refusal_threshold: float,
) -> bool:
    """Mirror of RefusalAblation.filter_fn."""
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if layer >= int(n_layers * (1.0 - prune_layer_percentage)):
        return True
    if kl_div_score > kl_threshold:
        return True
    if steering_score < induce_refusal_threshold:
        return True
    return False


def _run_refusal_ablation(
    input_checkpoint_path: str,
    output_checkpoint_path: str,
    out_dir: str,
    harmful_train_cfg: dict[str, Any],
    harmless_train_cfg: dict[str, Any],
    harmful_val_cfg: dict[str, Any],
    harmless_val_cfg: dict[str, Any],
    batch_size: int,
    refusal_strings: list[str],
    kl_threshold: float,
    prune_layer_percentage: float,
    filter_train: bool,
    filter_val: bool,
    max_layers: int | None,
    induce_refusal_threshold: float,
    cache_dir: str,
) -> None:
    """Run the full refusal ablation attack. Called via run_in_isolation.

    This mirrors the logic from RefusalAblation.run_attack, save_attacked_model,
    and log_attack_details, using the _iso_* helper functions instead of class methods.

    The expensive direction scoring is cached to ``cache_dir`` so subsequent trials
    (with different filtering hparams) can skip straight to direction selection.
    """
    from .configs import ModelConfig as RAModelConfig

    # --- Check cache ---
    cache_key = _compute_cache_key(
        input_checkpoint_path,
        harmful_train_cfg,
        harmless_train_cfg,
        harmful_val_cfg,
        harmless_val_cfg,
        batch_size,
        refusal_strings,
        filter_train,
        filter_val,
        max_layers,
    )
    cache_path = Path(cache_dir) / cache_key
    cached_scores_file = cache_path / "all_scores.json"
    cached_directions_file = cache_path / "candidate_directions.pt"
    cache_hit = cached_scores_file.exists() and cached_directions_file.exists()

    candidate_directions: Tensor | None = None
    json_output_all_scores: list[dict[str, Any]] = []

    if cache_hit:
        logger.info(f"Cache hit for refusal ablation scores at {cache_path}")
        with open(cached_scores_file) as f:
            json_output_all_scores = json.load(f)
        candidate_directions = torch.load(cached_directions_file, weights_only=True)
    else:
        logger.info(f"No cache found at {cache_path}, computing scores from scratch")

    # --- Load model (same as run_attack) ---
    model_config = RAModelConfig(
        name_or_path=Path(input_checkpoint_path),
        tokenizer_name_or_path=Path(input_checkpoint_path),
        download_dir=Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub",
    )
    model = WrappedHFModel(model_config)

    # --- Initialize layer modules (same as run_attack) ---
    block_modules = model.get_layers()
    attn_modules = model.get_attn_modules()
    mlp_modules = model.get_mlp_modules()

    if max_layers is not None and max_layers < len(block_modules):
        block_modules = block_modules[:max_layers]
        attn_modules = attn_modules[:max_layers]
        mlp_modules = mlp_modules[:max_layers]

    n_layers = len(block_modules)

    # --- Get refusal tokens (same as run_attack) ---
    tokenized_refusal_strings = model.tokenizer(refusal_strings, add_special_tokens=False).input_ids
    refusal_tokens = [tokens[0] for tokens in tokenized_refusal_strings]

    if not cache_hit:
        # --- Load datasets ---
        harmful_train = WrappedDataset(DatasetConfig(**harmful_train_cfg))
        harmless_train = WrappedDataset(DatasetConfig(**harmless_train_cfg))
        harmful_val = WrappedDataset(DatasetConfig(**harmful_val_cfg))
        harmless_val = WrappedDataset(DatasetConfig(**harmless_val_cfg))

        # --- Prepare datasets / filtering (same as prepare_attack) ---
        logger.info("Preparing datasets for refusal ablation attack")

        if filter_train:
            harmful_train_scores = _iso_get_refusal_scores(model, harmful_train, refusal_tokens, batch_size)
            harmless_train_scores = _iso_get_refusal_scores(model, harmless_train, refusal_tokens, batch_size)
            harmful_train_indices = torch.where(harmful_train_scores > 0)[0]
            harmless_train_indices = torch.where(harmless_train_scores < 0)[0]
            harmful_train.dataset = harmful_train.dataset.select(harmful_train_indices.tolist())
            harmless_train.dataset = harmless_train.dataset.select(harmless_train_indices.tolist())

        if filter_val:
            harmful_val_scores = _iso_get_refusal_scores(model, harmful_val, refusal_tokens, batch_size)
            harmless_val_scores = _iso_get_refusal_scores(model, harmless_val, refusal_tokens, batch_size)
            harmful_val_indices = torch.where(harmful_val_scores > 0)[0]
            harmless_val_indices = torch.where(harmless_val_scores < 0)[0]
            harmful_val.dataset = harmful_val.dataset.select(harmful_val_indices.tolist())
            harmless_val.dataset = harmless_val.dataset.select(harmless_val_indices.tolist())

        # --- Run the actual attack (same as run_attack) ---
        logger.info("Running refusal ablation attack")

        candidate_directions = _iso_generate_directions(
            model,
            harmful_train,
            harmless_train,
            block_modules,
            n_layers,
            batch_size,
        )

        n_pos, n_layer, _d_model = candidate_directions.shape

        ablation_kl_div_scores = torch.zeros((n_pos, n_layer), device=model.model.device, dtype=torch.float64)
        ablation_refusal_scores = torch.zeros((n_pos, n_layer), device=model.model.device, dtype=torch.float64)
        steering_refusal_scores = torch.zeros((n_pos, n_layer), device=model.model.device, dtype=torch.float64)

        json_output_all_scores.clear()

        # get logits for the harmless val set
        format_func = harmless_val.get_eval_formatter(model.tokenizer)
        baseline_harmless_logits = _iso_get_last_position_logits(model, harmless_val, format_func, batch_size)

        for source_pos in range(-n_pos, 0):
            # Create a tqdm instance for the inner loop
            pbar = tqdm(range(n_layer))
            for source_layer in pbar:
                # Update the progress bar description
                pbar.set_description(f"Processing position {source_pos} layer {source_layer}")

                ablation_dir = candidate_directions[source_pos, source_layer]
                fwd_pre_hooks, fwd_hooks = _iso_get_ablation_hooks(
                    block_modules,
                    attn_modules,
                    mlp_modules,
                    n_layers,
                    ablation_dir,
                )

                # 1. Compute KL divergence scores
                intervention_logits = _iso_get_last_position_logits(
                    model,
                    harmless_val,
                    format_func,
                    batch_size,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                )
                kl_div_score = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()
                ablation_kl_div_scores[source_pos, source_layer] = kl_div_score

                fwd_pre_hooks, fwd_hooks = _iso_get_ablation_hooks(
                    block_modules,
                    attn_modules,
                    mlp_modules,
                    n_layers,
                    ablation_dir,
                )

                # 2. Compute refusal ablation scores (reusing the same ablation hooks)
                refusal_scores = _iso_get_refusal_scores(
                    model,
                    harmful_val,
                    refusal_tokens,
                    batch_size,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                )
                refusal_score = refusal_scores.mean().item()
                ablation_refusal_scores[source_pos, source_layer] = refusal_score

                # 3. Compute refusal addition (steering) scores
                fwd_pre_hooks = [
                    (
                        block_modules[source_layer],
                        get_activation_addition_input_pre_hook(vector=ablation_dir, coeff=torch.tensor(1.0)),
                    )
                ]
                fwd_hooks = []
                refusal_scores = _iso_get_refusal_scores(
                    model,
                    harmless_val,
                    refusal_tokens,
                    batch_size,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                )
                steering_score = refusal_scores.mean().item()
                steering_refusal_scores[source_pos, source_layer] = steering_score

                # 4. Record scores
                json_output_all_scores.append(
                    {
                        "position": source_pos,
                        "layer": source_layer,
                        "refusal_score": refusal_scores.mean().item(),
                        "steering_score": steering_score,
                        "kl_div_score": kl_div_score,
                    }
                )

        # --- Save to cache ---
        cache_path.mkdir(parents=True, exist_ok=True)
        with open(cached_scores_file, "w") as f:
            json.dump(json_output_all_scores, f, indent=4)
        torch.save(candidate_directions.cpu(), cached_directions_file)
        logger.info(f"Cached scores and directions to {cache_path}")

    # --- Filter and select best direction (uses sweep hparams) ---
    n_pos, n_layer = candidate_directions.shape[0], candidate_directions.shape[1]  # pyright: ignore[reportOptionalMemberAccess]

    json_output_filtered_scores: list[dict[str, Any]] = []
    filtered_scores: list[tuple[float, int, int]] = []

    for entry in json_output_all_scores:
        refusal_score = entry["refusal_score"]
        steering_score = entry["steering_score"]
        kl_div_score = entry["kl_div_score"]
        source_layer = entry["layer"]

        sorting_score = -refusal_score

        discard_direction = _iso_filter_fn(
            refusal_score=refusal_score,
            steering_score=steering_score,
            kl_div_score=kl_div_score,
            layer=source_layer,
            n_layers=n_layers,
            prune_layer_percentage=prune_layer_percentage,
            kl_threshold=kl_threshold,
            induce_refusal_threshold=induce_refusal_threshold,
        )

        if not discard_direction:
            filtered_scores.append((sorting_score, entry["position"], source_layer))
            json_output_filtered_scores.append(entry)

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x["refusal_score"], reverse=False)
    # Sort filtered scores in descending order
    filtered_scores.sort(key=lambda x: x[0], reverse=True)

    if not filtered_scores:
        # No directions passed filtering — fall back to best unfiltered direction
        logger.warning("No directions passed filtering criteria. Using best unfiltered direction.")
        best_unfiltered = min(json_output_all_scores, key=lambda x: x["refusal_score"])
        pos = best_unfiltered["position"]
        layer = best_unfiltered["layer"]
    else:
        _score, pos, layer = filtered_scores[0]

    best_direction = candidate_directions[pos, layer]

    # apply the best direction to the model
    model.orthogonalize_weights(best_direction)

    # --- Save attacked model (same as save_attacked_model) ---
    Path(output_checkpoint_path).mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(output_checkpoint_path)
    logger.info(f"Refusal ablated model saved to {output_checkpoint_path}")

    model.tokenizer.save_pretrained(output_checkpoint_path)
    logger.info(f"Tokenizer saved to {output_checkpoint_path}")

    # Patch head_dim into config.json if missing — vLLM requires it explicitly
    # for Llama-family architectures (including Mistral) and will fail with a
    # TypeError if the field is absent or null.
    config_path = Path(output_checkpoint_path) / "config.json"
    with open(config_path) as f:
        saved_config = json.load(f)
    if saved_config.get("head_dim") is None:
        hidden_size = saved_config.get("hidden_size", 0)
        num_heads = saved_config.get("num_attention_heads", 0)
        if hidden_size and num_heads:
            saved_config["head_dim"] = hidden_size // num_heads
            with open(config_path, "w") as f:
                json.dump(saved_config, f, indent=2)
            logger.info(f"Patched head_dim={saved_config['head_dim']} into {config_path}")

    # --- Log attack details (same as log_attack_details) ---
    if filtered_scores:
        score, pos, layer = filtered_scores[0]
    else:
        best_unfiltered = min(json_output_all_scores, key=lambda x: x["refusal_score"])
        score = best_unfiltered["refusal_score"]
        pos = best_unfiltered["position"]
        layer = best_unfiltered["layer"]

    logger.info(f"Selected direction: position {pos}, layer {layer}, score {score}")

    # Save metadata to output directory
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "all_scores.json", "w") as f:
        json.dump(json_output_all_scores, f, indent=4)
    logger.info(f"All scores saved to {output_dir / 'all_scores.json'}")

    with open(output_dir / "filtered_scores.json", "w") as f:
        json.dump(json_output_filtered_scores, f, indent=4)
    logger.info(f"Filtered scores saved to {output_dir / 'filtered_scores.json'}")

    torch.save(best_direction, output_dir / "best_direction.pt")
    logger.info(f"Best direction saved to {output_dir / 'best_direction.pt'}")
