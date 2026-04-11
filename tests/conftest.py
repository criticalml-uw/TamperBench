"""Pytest configuration and fixtures."""

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip gpu_required tests when no GPU is available."""
    del config  # unused
    if torch.cuda.is_available():
        return

    skip_no_gpu = pytest.mark.skip(reason="requires GPU (CUDA not available)")
    for item in items:
        if "gpu_required" in item.keywords:
            item.add_marker(skip_no_gpu)
