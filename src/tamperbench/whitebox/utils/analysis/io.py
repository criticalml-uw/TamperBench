"""I/O utilities for analysis file operations.

Provides JSON loading/writing, CSV export, directory management,
and discovery functions for results directories.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def write_json(
    data: dict[str, Any] | list[Any],
    path: Path,
    indent: int = 2,
) -> None:
    """Write data to JSON file, creating parent directories as needed.

    Args:
        data: Dictionary or list to serialize as JSON
        path: Output file path
        indent: JSON indentation level (default: 2)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=indent)

    LOGGER.info(f"Wrote JSON to {path}")


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, returning the path for chaining.

    Args:
        path: Directory path to create

    Returns:
        The same path, for method chaining

    Example:
        >>> output_dir = ensure_dir(Path("results/aggregated"))
        >>> (output_dir / "heatmap.json").write_text("{}")
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_models(results_dir: Path) -> list[str]:
    """Find all model directories in results directory.

    Excludes directories starting with 'aggregated', 'epsilon_', or 'configs_'.

    Args:
        results_dir: Parent results directory to search

    Returns:
        Sorted list of model directory names

    Example:
        >>> find_models(Path("results/sweep_001"))
        ['llama-8b', 'mistral-7b', 'qwen-7b']
    """
    if not results_dir.exists():
        return []

    model_dirs = [
        d.name
        for d in results_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith("aggregated")
        and not d.name.startswith("epsilon_")
        and not d.name.startswith("configs_")
    ]

    return sorted(model_dirs)
