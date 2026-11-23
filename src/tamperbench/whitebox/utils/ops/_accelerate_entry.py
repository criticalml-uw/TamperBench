"""Entry point run in each process by `run_with_accelerate`. Not meant to be called directly.

Usage (done by `run_with_accelerate`): ``accelerate launch ... -m tamperbench.whitebox.utils.ops._accelerate_entry <payload.pkl>``
"""

import functools
import importlib
import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch.distributed as dist


def main() -> None:
    """Load the pickled target and arguments, then call the target."""
    payload_path = Path(sys.argv[1])
    with payload_path.open("rb") as f:
        module_name, qualname, args = pickle.load(f)

    target: Callable[..., Any] = functools.reduce(getattr, qualname.split("."), importlib.import_module(module_name))
    try:
        target(*args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
