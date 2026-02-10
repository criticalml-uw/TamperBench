"""Training utilities for CRL defense."""

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from tamperbench.whitebox.defenses.crl.config import CRLConfig


def cosine_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute cosine distance: 1 - cosine_similarity.

    Following the CRL author's recommendation to use cosine distance
    over L2 or mixed distance.
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    return 1 - (x_norm * y_norm).sum(dim=-1)


@dataclass
class TrainingState:
    """Holds the current training state for checkpointing."""

    step: int
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, torch.Tensor]
    scaler_state_dict: dict[str, torch.Tensor]


def extract_representations(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: tuple[int, ...],
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract hidden state representations from specified layers.

    Representations are extracted from ALL tokens in the assistant response,
    matching the CRL paper's approach (also used by RepBend and Circuit Breaking).
    The response_mask indicates which positions are response tokens (1) vs
    request/padding tokens (0).

    Args:
        model: The model to extract from.
        input_ids: Input token IDs of shape (batch, seq_len).
        attention_mask: Attention mask of shape (batch, seq_len).
        layers: Layer indices to extract from.
        response_mask: Mask indicating response token positions, shape (batch, seq_len).
            1 for response tokens, 0 for request/padding.

    Returns:
        tuple of:
            - torch.Tensor: Hidden states, shape (num_layers, batch, seq_len, hidden_dim)
            - torch.Tensor: Target mask for loss computation, shape (num_layers, batch, seq_len, 1)
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # Get hidden states from specified layers
    hidden_states = outputs.hidden_states
    layer_hiddens = []

    for layer_idx in layers:
        if layer_idx < len(hidden_states):
            # Shape: (batch, seq_len, hidden_dim)
            layer_hiddens.append(hidden_states[layer_idx])

    if not layer_hiddens:
        raise ValueError(
            f"No hidden states extracted: requested layers {layers} but model only has "
            f"{len(hidden_states)} layers. Set representation_layers in CRLConfig accordingly."
        )

    # Stack: (num_layers, batch, seq_len, hidden_dim)
    stacked_hidden = torch.stack(layer_hiddens, dim=0)

    # Create target mask from response_mask
    # Shape: (batch, seq_len) -> (num_layers, batch, seq_len, 1)
    num_layers = len(layer_hiddens)

    if response_mask is not None:
        target_mask = response_mask.float().unsqueeze(0).unsqueeze(-1)
        target_mask = target_mask.expand(num_layers, -1, -1, -1)
    else:
        # Fallback: use all non-padding tokens (last-token-like behavior).
        # This is likely a data pipeline issue -- the CRL paper author warns that
        # last-token mode "probably never [captures anything] very harmful" and
        # "heavily impacted performance."
        warnings.warn(
            "response_mask is None; falling back to attention_mask (all non-padding tokens). "
            "This likely indicates a dataset preprocessing issue. CRL performance may be degraded.",
            stacklevel=2,
        )
        target_mask = attention_mask.float().unsqueeze(0).unsqueeze(-1)
        target_mask = target_mask.expand(num_layers, -1, -1, -1)

    return stacked_hidden, target_mask


def compute_triplet_loss(
    h_benign_orig: torch.Tensor,
    h_benign_new: torch.Tensor,
    h_harmful_orig: torch.Tensor,
    h_harmful_new: torch.Tensor,
    mask_benign: torch.Tensor,
    mask_harmful: torch.Tensor,
    config: CRLConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute triplet losses for benign and harmful samples.

    From the paper (Equations 7-9):
    - Benign: keep new representations close to original, far from harmful mean
    - Harmful: push new representations close to harmful mean, far from original

    Now operates on full sequence hidden states with masks to select response tokens.

    Args:
        h_benign_orig: Original model benign hidden states (num_layers, batch, seq_len, hidden_dim).
        h_benign_new: New model benign hidden states (num_layers, batch, seq_len, hidden_dim).
        h_harmful_orig: Original model harmful hidden states (num_layers, batch, seq_len, hidden_dim).
        h_harmful_new: New model harmful hidden states (num_layers, batch, seq_len, hidden_dim).
        mask_benign: Mask for benign response tokens (num_layers, batch, seq_len, 1).
        mask_harmful: Mask for harmful response tokens (num_layers, batch, seq_len, 1).
        config: CRL configuration with distance weights and margins.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Benign and harmful triplet losses.
    """
    # Apply masks to hidden states (zero out non-response positions).
    # NOTE: Distance is computed on the full sequence (including zeroed positions) for
    # simplicity. This is functionally correct because the loss at zeroed positions is
    # multiplied by the mask below, but is computationally wasteful for long sequences
    # with short responses. Acceptable for now given typical response lengths.
    h_benign_orig_masked = h_benign_orig * mask_benign
    h_benign_new_masked = h_benign_new * mask_benign
    h_harmful_orig_masked = h_harmful_orig * mask_harmful
    h_harmful_new_masked = h_harmful_new * mask_harmful

    # Compute harmful centroid: mean of harmful representations across batch and sequence.
    # DESIGN NOTE: The centroid is computed from the *training* model's representations
    # (h_harmful_new), not the frozen model's. This is intentional -- the centroid defines
    # the cluster center in the evolving representation space, so it must track the
    # training model. Using the frozen model's representations would create a fixed target
    # that doesn't reflect where the training model's harmful representations actually are.
    # The centroid is per-batch (no momentum/EMA); for small batch sizes this may be noisy,
    # but matches the reference CRL implementation.
    h_harmful_mean = h_harmful_new_masked.sum(dim=(1, 2), keepdim=True) / (
        mask_harmful.sum(dim=(1, 2), keepdim=True) + 1e-8
    )

    # Benign triplet: d(h_b, h'_b) - d(h'_b, h_h_mean) + margin
    # Goal: keep benign representations close to original, far from harmful mean
    d_benign_pos = cosine_dist(h_benign_orig_masked, h_benign_new_masked)
    d_benign_neg = cosine_dist(h_benign_new_masked, h_harmful_mean.expand_as(h_benign_new_masked))

    # Compute loss only at masked positions and average
    loss_benign_raw = F.relu(d_benign_pos - d_benign_neg + config.margin_benign)
    loss_benign_raw = loss_benign_raw * mask_benign.squeeze(-1)
    loss_benign = loss_benign_raw.sum() / (mask_benign.sum() + 1e-8)

    # Harmful triplet: d(h'_h, h_h_mean) - d(h'_h, h_h) + margin
    # Goal: push harmful representations toward cluster mean, away from original
    d_harmful_pos = cosine_dist(h_harmful_new_masked, h_harmful_mean.expand_as(h_harmful_new_masked))
    d_harmful_neg = cosine_dist(h_harmful_new_masked, h_harmful_orig_masked)

    loss_harmful_raw = F.relu(d_harmful_pos - d_harmful_neg + config.margin_harmful)
    loss_harmful_raw = loss_harmful_raw * mask_harmful.squeeze(-1)
    loss_harmful = loss_harmful_raw.sum() / (mask_harmful.sum() + 1e-8)

    return loss_benign, loss_harmful


def compute_kl_divergence_loss(
    frozen_model: PreTrainedModel,
    trainable_model: PeftModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence loss to preserve benign behavior.

    TODO: This runs a redundant forward pass for both models (the same inputs were
    already processed by extract_representations for hidden states). A future
    optimization would return logits from extract_representations and reuse them here.

    Args:
        frozen_model: Original frozen model for reference.
        trainable_model: Trainable model with LoRA adapters.
        input_ids: Input token IDs.
        attention_mask: Attention mask.

    Returns:
        torch.Tensor: KL divergence loss (scalar).
    """
    with torch.no_grad():
        orig_logits = frozen_model(input_ids, attention_mask=attention_mask).logits

    new_logits = trainable_model(input_ids, attention_mask=attention_mask).logits

    # KL divergence: D_KL(orig || new), masked to non-padding positions only.
    # Without masking, padding positions (where logits are noise) would dominate
    # the loss for short sequences padded to max_length.
    kl_per_position = F.kl_div(
        F.log_softmax(new_logits, dim=-1),
        F.softmax(orig_logits, dim=-1),
        reduction="none",
    ).sum(dim=-1)  # sum over vocab -> (batch, seq_len)

    mask = attention_mask.float()
    return (kl_per_position * mask).sum() / (mask.sum() + 1e-8)


def _create_data_iterators(
    benign_loader: DataLoader[Any],
    harmful_loader: DataLoader[Any],
) -> tuple[Iterator[dict[str, torch.Tensor]], Iterator[dict[str, torch.Tensor]]]:
    """Create iterators for data loaders.

    Args:
        benign_loader: DataLoader for benign samples.
        harmful_loader: DataLoader for harmful samples.

    Returns:
        Tuple of iterators for benign and harmful data.
    """
    return iter(benign_loader), iter(harmful_loader)


def _get_next_batch(
    data_iter: Iterator[dict[str, torch.Tensor]],
    data_loader: DataLoader[Any],
) -> tuple[dict[str, torch.Tensor], Iterator[dict[str, torch.Tensor]]]:
    """Get next batch, cycling the iterator if exhausted.

    Args:
        data_iter: Current iterator.
        data_loader: DataLoader to recreate iterator from.

    Returns:
        Tuple of (batch, updated_iterator).
    """
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        batch = next(data_iter)
    return batch, data_iter


def load_checkpoint(
    checkpoint_path: Path,
    trainable_model: PeftModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,  # pyright: ignore[reportPrivateImportUsage]
) -> int:
    """Load training state from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        trainable_model: Model to load weights into.
        optimizer: Optimizer to load state into.
        scaler: AMP scaler to load state into.

    Returns:
        int: Step number to resume from.
    """
    print(f"   Resuming from checkpoint: {checkpoint_path.name}")
    # Load to CPU first to avoid GPU memory fragmentation
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    trainable_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_step = ckpt["step"] + 1
    del ckpt  # Free CPU memory
    torch.cuda.empty_cache()
    print(f"   Resuming from step {start_step}")
    return start_step


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    trainable_model: PeftModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,  # pyright: ignore[reportPrivateImportUsage]
) -> Path:
    """Save training checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        step: Current training step.
        trainable_model: Model to save.
        optimizer: Optimizer to save.
        scaler: AMP scaler to save.

    Returns:
        Path: Path to saved checkpoint.
    """
    ckpt_path = checkpoint_dir / f"step_{step:05d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": trainable_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def run_training_loop(
    frozen_model: PreTrainedModel,
    trainable_model: PeftModel,
    benign_loader: DataLoader[Any],
    harmful_loader: DataLoader[Any],
    config: CRLConfig,
    output_path: Path,
) -> None:
    """Run the CRL training loop.

    Args:
        frozen_model: Frozen reference model.
        trainable_model: Trainable model with LoRA adapters.
        benign_loader: DataLoader for benign samples.
        harmful_loader: DataLoader for harmful samples.
        config: CRL configuration.
        output_path: Path for saving checkpoints.
    """
    optimizer = torch.optim.AdamW(
        trainable_model.parameters(),
        lr=config.learning_rate,
    )

    # Use AMP for faster training
    scaler = torch.amp.GradScaler()  # pyright: ignore[reportPrivateImportUsage]

    trainable_model.train()
    layers = config.representation_layers

    # Set up checkpoint directory
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0

    # Check for existing checkpoint to resume from
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if checkpoints:
        start_step = load_checkpoint(checkpoints[-1], trainable_model, optimizer, scaler)

    # Create data iterators
    benign_iter, harmful_iter = _create_data_iterators(benign_loader, harmful_loader)

    pbar = tqdm(range(start_step, config.num_steps), desc="Training")
    for step in pbar:
        optimizer.zero_grad()

        # Get batches (cycling if needed)
        benign_batch, benign_iter = _get_next_batch(benign_iter, benign_loader)
        harmful_batch, harmful_iter = _get_next_batch(harmful_iter, harmful_loader)

        # Move to device
        device = cast(torch.device, trainable_model.device)
        benign_ids = benign_batch["input_ids"].to(device)
        benign_mask = benign_batch["attention_mask"].to(device)
        harmful_ids = harmful_batch["input_ids"].to(device)
        harmful_mask = harmful_batch["attention_mask"].to(device)

        # Get response masks (for selecting which tokens to use in loss)
        benign_response_mask = benign_batch.get("response_mask")
        harmful_response_mask = harmful_batch.get("response_mask")
        if benign_response_mask is not None:
            benign_response_mask = benign_response_mask.to(device)
        if harmful_response_mask is not None:
            harmful_response_mask = harmful_response_mask.to(device)

        # Use autocast for mixed precision training.
        # NOTE: autocast covers both frozen and trainable model forward passes. The frozen
        # model runs in float16 under autocast, which is acceptable since both models see
        # the same precision and the relative distances are what matter for the loss.
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  # pyright: ignore[reportPrivateImportUsage]
            # Get original representations (frozen model)
            with torch.no_grad():
                h_benign_orig, mask_benign = extract_representations(
                    frozen_model,
                    benign_ids,
                    benign_mask,
                    layers,
                    response_mask=benign_response_mask,
                )
                h_harmful_orig, mask_harmful = extract_representations(
                    frozen_model,
                    harmful_ids,
                    harmful_mask,
                    layers,
                    response_mask=harmful_response_mask,
                )

            # Get new representations (trainable model)
            # PeftModel wraps PreTrainedModel but isn't a subtype in the stubs
            h_benign_new, _ = extract_representations(
                trainable_model,  # pyright: ignore[reportArgumentType]
                benign_ids,
                benign_mask,
                layers,
                response_mask=benign_response_mask,
            )
            h_harmful_new, _ = extract_representations(
                trainable_model,  # pyright: ignore[reportArgumentType]
                harmful_ids,
                harmful_mask,
                layers,
                response_mask=harmful_response_mask,
            )

            # Compute triplet losses (mean is computed inside compute_triplet_loss now)
            loss_benign, loss_harmful = compute_triplet_loss(
                h_benign_orig,
                h_benign_new,
                h_harmful_orig,
                h_harmful_new,
                mask_benign,
                mask_harmful,
                config,
            )

            # Compute KL divergence on benign logits
            loss_kl = compute_kl_divergence_loss(frozen_model, trainable_model, benign_ids, benign_mask)

            # Total loss
            loss = config.alpha * loss_benign + config.beta * loss_harmful + config.gamma * loss_kl

        # Backward pass with gradient clipping (using scaler for AMP)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "benign": f"{loss_benign.item():.4f}",
                "harmful": f"{loss_harmful.item():.4f}",
                "kl": f"{loss_kl.item():.4f}",
            }
        )

        # Save checkpoint periodically
        if (step + 1) % config.checkpoint_interval == 0:
            ckpt_path = save_checkpoint(checkpoint_dir, step, trainable_model, optimizer, scaler)
            print(f"\n   Checkpoint saved: {ckpt_path.name}")

    # Save final checkpoint if not already saved at the last interval
    if config.num_steps % config.checkpoint_interval != 0:
        ckpt_path = save_checkpoint(checkpoint_dir, config.num_steps - 1, trainable_model, optimizer, scaler)
        print(f"\n   Final checkpoint saved: {ckpt_path.name}")
