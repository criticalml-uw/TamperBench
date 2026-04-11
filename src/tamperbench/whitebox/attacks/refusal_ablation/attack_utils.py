# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportUnusedCallResult=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportUnusedParameter=false
# ruff: noqa: F722
"""Utility functions for activation hooks and KL divergence in refusal ablation attack.

Adapted from:
https://github.com/AlignmentResearch/safety_gap/attack/utils.py
"""

import contextlib
import functools
from collections.abc import Callable
from typing import Any

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: list[tuple[torch.nn.Module, Callable[..., Any]]],
    module_forward_hooks: list[tuple[torch.nn.Module, Callable[..., Any]]],
    **kwargs,
):
    """Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    **kwargs
        Additional keyword arguments passed to hooks
    """
    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_direction_ablation_input_pre_hook(direction: Tensor):
    """Create a pre-forward hook that ablates a direction from input activations.

    Args:
        direction: The direction vector to ablate from activations

    Returns:
        A hook function that removes the specified direction from activations
    """
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"]

        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input

        d = direction.to(activation)
        activation -= (activation @ d).unsqueeze(-1) * d

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn


def get_direction_ablation_output_hook(direction: Tensor):
    """Create a forward hook that ablates a direction from output activations.

    Args:
        direction: The direction vector to ablate from activations

    Returns:
        A hook function that removes the specified direction from activations
    """
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

    def hook_fn(module, input, output):
        activation: Float[Tensor, "batch_size seq_len d_model"]
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output

        d = direction.to(activation)
        activation -= (activation @ d).unsqueeze(-1) * d

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def get_mean_activations_pre_hook(layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: list[int]):
    """Create a pre-forward hook that accumulates mean activations.

    Args:
        layer: The layer index to cache activations for
        cache: Tensor to store accumulated activations
        n_samples: Number of samples to average over
        positions: Token positions to extract activations from

    Returns:
        A hook function that accumulates mean activations into the cache
    """

    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)

    return hook_fn


def get_activation_addition_input_pre_hook(
    vector: Float[Tensor, "d_model"],
    coeff: Float[Tensor, ""],
):
    """Create a pre-forward hook that adds a scaled vector to input activations.

    Args:
        vector: The vector to add to activations
        coeff: Scaling coefficient for the vector

    Returns:
        A hook function that adds the scaled vector to activations
    """

    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"]

        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input

        v = vector.to(activation)
        activation += coeff * v

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn


def masked_mean(seq, mask=None, dim=1, keepdim=False):
    """Compute mean of a sequence with optional masking.

    Args:
        seq: Input sequence tensor
        mask: Optional boolean mask tensor
        dim: Dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension

    Returns:
        Mean of the sequence, optionally masked
    """
    if mask is None:
        return seq.mean(dim=dim)

    if seq.ndim == 3:
        mask = rearrange(mask, "b n -> b n 1")

    masked_seq = seq.masked_fill(~mask, 0.0)
    numer = masked_seq.sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim)

    masked_mean = numer / denom.clamp(min=1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.0)
    return masked_mean


def kl_div_fn(
    logits_a: Float[Tensor, "batch seq_pos d_vocab"],
    logits_b: Float[Tensor, "batch seq_pos d_vocab"],
    mask: Int[Tensor, "batch seq_pos"] | None = None,
    epsilon: float = 1e-6,
) -> Float[Tensor, "batch"]:
    """Compute the KL divergence loss between two tensors of logits.

    Args:
        logits_a: First set of logits
        logits_b: Second set of logits
        mask: Optional mask for valid positions
        epsilon: Small constant for numerical stability

    Returns:
        KL divergence per batch element
    """
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        return masked_mean(kl_divs, mask).mean(dim=-1)
