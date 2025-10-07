"""Utilities for reading configs and persisting Optuna sweep artifacts."""

from __future__ import annotations

# pyright: reportAny=false
from pathlib import Path
from typing import Any

import yaml


def yaml_to_dict(yaml_file_path: Path) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    """Read a YAML file and convert its content into a Python dictionary.

    Args:
        yaml_file_path: The path to the YAML file.

    Returns:
        dict[str, Any]: Parsed YAML payload.
    """
    with open(yaml_file_path) as file:
        data = yaml.safe_load(file)
    return data
