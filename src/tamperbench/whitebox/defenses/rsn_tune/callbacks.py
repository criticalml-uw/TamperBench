"""Training callbacks for RSN-Tune defense."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel, TrainerCallback


def make_mask_hook(mask: torch.Tensor) -> Any:
    """Create gradient hook that applies mask, with dtype caching."""
    cache: list[torch.Tensor | None] = [None]

    def hook(grad: torch.Tensor) -> torch.Tensor | None:
        if grad is None:
            return None
        if cache[0] is None or cache[0].dtype != grad.dtype:
            cache[0] = mask.to(dtype=grad.dtype, device=grad.device)
        return grad * cache[0]

    return hook


class GradientMaskCallback(TrainerCallback):  # type: ignore[misc]
    """Masks gradients during training to only update specified neurons.

    Applies gradient masks to parameters with selected neurons, zeroing gradients
    for non-selected dimensions. Parameters without any selected neurons should
    have requires_grad=False set before training to save optimizer memory.

    Args:
        model: The model being trained.
        masks: Dict mapping parameter names to mask tensors. Masks should be float
            tensors with 1.0 for neurons to train and 0.0 for neurons to freeze.
    """

    model: PreTrainedModel
    masks: dict[str, torch.Tensor]
    hooks: list[Any]

    def __init__(
        self,
        model: PreTrainedModel,
        masks: dict[str, torch.Tensor],
    ) -> None:
        """Initialize the gradient mask callback."""
        super().__init__()
        self.model = model
        self.masks = masks
        self.hooks = []

    def on_train_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        """Register gradient hooks at the start of training."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.masks:
                # Partial mask: only allow gradients for selected neurons
                self.hooks.append(param.register_hook(make_mask_hook(self.masks[name])))

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Remove gradient hooks at the end of training."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
