"""Hooks for the latent attack evaluation: https://github.com/aengusl/latent-adversarial-training/."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
from typing_extensions import override

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer  # pyright: ignore[reportMissingImports]

_is_using_tl = False

try:
    from transformer_lens import HookedTransformer  # pyright: ignore[reportMissingImports]

    _is_using_tl = True
except ImportError:
    pass


class CustomHook(torch.nn.Module):
    """A wrapper module that applies a custom hook function during the forward pass."""

    def __init__(self, module: torch.nn.Module, hook_fn: Callable[..., Any]) -> None:
        """Initialize the CustomHook.

        Args:
            module: The wrapped PyTorch module.
            hook_fn: A function applied to the module outputs when enabled.
        """
        super().__init__()
        self.module: torch.nn.Module = module
        self.hook_fn: Callable[..., Any] = hook_fn
        self.enabled: bool = True

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the hook function to the wrapped module's output if enabled."""
        if self.enabled:
            return self.hook_fn(self.module(*args, **kwargs))
        else:
            return self.module(*args, **kwargs)


class TLHook(torch.nn.Module):
    """A wrapper module for transformer-lens hooks."""

    def __init__(self, hook_fn: Callable[..., Any]) -> None:
        """Initialize the TLHook.

        Args:
            hook_fn: A function applied to activations when enabled.
        """
        super().__init__()
        self.hook_fn: Callable[..., Any] = hook_fn
        self.enabled: bool = True

    @override
    def forward(self, activations: Any, hook: Any) -> Any:
        """Apply the hook function to activations if enabled."""
        if self.enabled:
            return self.hook_fn(activations)
        else:
            return activations


def _remove_hook(parent: torch.nn.Module, target: str) -> None:
    """Remove a CustomHook from the given parent module if present."""
    for name, module in parent.named_children():
        if name == target:
            setattr(parent, name, module.module)
            return


def insert_hook(parent: torch.nn.Module, target: str, hook_fn: Callable[..., Any]) -> CustomHook:
    """Insert a CustomHook into the given parent module at submodule `target`."""
    hook = None
    for name, module in parent.named_children():
        if name == target and hook is None:
            hook = CustomHook(module, hook_fn)
            setattr(parent, name, hook)
        elif name == target and hook is not None:
            _remove_hook(parent, target)
            raise ValueError(f"Multiple modules with name {target} found, removed hooks")

    if hook is None:
        raise ValueError(f"No module with name {target} found")

    return hook


def remove_hook(parent: torch.nn.Module, target: str) -> None:
    """Remove a previously inserted CustomHook from the given parent module."""
    is_removed = False
    for name, module in parent.named_children():
        if name == target and isinstance(module, CustomHook):
            setattr(parent, name, module.module)
            is_removed = True
        elif name == target and not isinstance(module, CustomHook):
            raise ValueError(f"Module {target} is not a hook")
        elif name == target:
            raise ValueError(f"FATAL: Multiple modules with name {target} found")

    if not is_removed:
        raise ValueError(f"No module with name {target} found")


def clear_hooks(model: torch.nn.Module) -> None:
    """Recursively remove all CustomHooks from the given model."""
    for name, module in model.named_children():
        if isinstance(module, CustomHook):
            setattr(model, name, module.module)
            clear_hooks(module.module)
        else:
            clear_hooks(module)


def add_hooks(
    model: HookedTransformer | torch.nn.Module,
    create_adversary: Callable[[tuple[int, str] | tuple[str, str]], Any],
    adversary_locations: list[tuple[int, str]] | list[tuple[str, str]],
) -> tuple[list[Any], list[Any]]:
    """Add adversarial hooks to a model.

    Args:
        model: A HookedTransformer or torch.nn.Module.
        create_adversary: Function that creates adversary modules.
        adversary_locations: List of (layer, subcomponent) locations for hooks.

    Returns:
        Tuple of (adversaries, hooks) inserted into the model.
    """
    if _is_using_tl:
        is_tl_model = isinstance(model, HookedTransformer)
    else:
        is_tl_model = False

    adversaries: list[Any] = []
    hooks: list[Any] = []

    if is_tl_model and isinstance(adversary_locations[0][0], int):
        tl_model = cast("HookedTransformer", model)
        int_locations = cast(list[tuple[int, str]], adversary_locations)
        for layer, subcomponent in int_locations:
            hook = tl_model.blocks[layer].get_submodule(subcomponent)
            adversaries.append(create_adversary((layer, subcomponent)))
            hooks.append(TLHook(adversaries[-1]))
            hook.add_hook(hooks[-1])

        return adversaries, hooks

    if len(adversary_locations) == 0:
        raise ValueError("No hook points provided")

    str_locations = cast(list[tuple[str, str]], adversary_locations)
    for layer, subcomponent in str_locations:
        parent = model.get_submodule(layer)
        adversaries.append(create_adversary((layer, subcomponent)))
        hooks.append(insert_hook(parent, subcomponent, adversaries[-1]))

    return adversaries, hooks


class AdversaryWrapper(torch.nn.Module):
    """A wrapper module that passes outputs through an adversary."""

    def __init__(self, module: torch.nn.Module, adversary: Any) -> None:
        """Initialize the AdversaryWrapper.

        Args:
            module: The wrapped PyTorch module.
            adversary: A function or module applied to the module outputs.
        """
        super().__init__()
        self.module: torch.nn.Module = module
        self.adversary: Any = adversary

    @override
    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """Forward pass with adversarial modification of outputs."""
        outputs = self.module(*inputs, **kwargs)
        return self.adversary(outputs)


def deepspeed_add_hooks(
    model: torch.nn.Module,
    create_adversary: Callable[[tuple[int, str] | tuple[str, str]], Any],
    adversary_locations: list[tuple[int, str]] | list[tuple[str, str]],
) -> tuple[list[Any], list[Any]]:
    """Add adversarial hooks to a DeepSpeed model.

    Args:
        model: A PyTorch model.
        create_adversary: Function that creates adversary modules.
        adversary_locations: List of (layer, subcomponent) locations for hooks.

    Returns:
        Tuple of (adversaries, hooks) inserted into the model.
    """
    adversaries: list[Any] = []
    hooks: list[Any] = []

    if len(adversary_locations) == 0:
        raise ValueError("No hook points provided")

    str_locations = cast(list[tuple[str, str]], adversary_locations)
    for layer, subcomponent in str_locations:
        parent = model.get_submodule(layer)
        adversary = create_adversary((layer, subcomponent))
        adversaries.append(adversary)
        submodule = parent.get_submodule(subcomponent)
        wrapped_module = AdversaryWrapper(submodule, adversary)
        hooks.append(wrapped_module)
        setattr(parent, subcomponent, wrapped_module)

    return adversaries, hooks
