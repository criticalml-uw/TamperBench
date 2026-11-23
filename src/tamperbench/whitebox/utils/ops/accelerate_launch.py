"""Utilities for running a function under `accelerate launch` (e.g., multi-GPU FSDP training)."""

import pickle
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

_ENTRY_MODULE = "tamperbench.whitebox.utils.ops._accelerate_entry"


def fsdp_launch_args(num_processes: int) -> list[str]:
    """Build `accelerate launch` arguments for single-node FSDP training.

    Each process (one per GPU) holds a shard of every layer, and all processes train in parallel. Settings:

    - Full sharding of parameters, gradients and optimizer states, wrapped per transformer block.
    - Full (unsharded) state dict on save, so the saved checkpoint is a normal Hugging Face checkpoint.
    - No accelerate mixed precision: models are loaded in bf16, and enabling mixed precision would upcast
      the weights to fp32.

    Args:
        num_processes: Number of processes to launch, normally one per GPU.

    Returns:
        list[str]: Arguments to pass to `run_with_accelerate` as `launch_args`.
    """
    return [
        "--num_machines",
        "1",
        "--num_processes",
        str(num_processes),
        "--mixed_precision",
        "no",
        "--dynamo_backend",
        "no",
        "--use_fsdp",
        "--fsdp_sharding_strategy",
        "FULL_SHARD",
        "--fsdp_auto_wrap_policy",
        "TRANSFORMER_BASED_WRAP",
        "--fsdp_backward_prefetch",
        "BACKWARD_PRE",
        "--fsdp_state_dict_type",
        "FULL_STATE_DICT",
        "--fsdp_use_orig_params",
        "true",
        "--fsdp_sync_module_states",
        "true",
        "--fsdp_cpu_ram_efficient_loading",
        "false",
        "--fsdp_offload_params",
        "false",
    ]


def run_with_accelerate(
    target: Callable[..., Any],
    args: tuple[Any, ...] = (),
    *,
    launch_args: list[str],
    error_context: str = "accelerate launch",
) -> None:
    """Run `target(*args)` in every process started by `accelerate launch`.

    `accelerate launch` starts a script rather than a Python function, so the target's import path and its
    arguments are pickled to a temporary file, and a small entry module loads them in each process and calls
    the target. Like `run_in_isolation`, the caller's process does not hold GPU memory used by the target,
    but return values are not passed back.

    Args:
        target: A module-level function (it must be importable by name). Called in every process.
        args: Positional arguments for `target`. Must be picklable.
        launch_args: Arguments for `accelerate launch` (e.g. from `fsdp_launch_args`).
        error_context: Context string for error messages (e.g., "Full parameter finetune attack").

    Raises:
        ValueError: If `target` is not importable by name (e.g., a lambda or nested function).
        RuntimeError: If the launched processes exit with a non-zero code.
    """
    if "<" in target.__qualname__:
        raise ValueError(f"{error_context}: target must be a module-level function, got {target.__qualname__}")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump((target.__module__, target.__qualname__, args), f)
        payload_path = Path(f.name)

    cmd = [sys.executable, "-m", "accelerate.commands.launch", *launch_args, "-m", _ENTRY_MODULE, str(payload_path)]
    try:
        result = subprocess.run(cmd, check=False)
    finally:
        payload_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"{error_context} failed under accelerate launch (exit code {result.returncode})")
