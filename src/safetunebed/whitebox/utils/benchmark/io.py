"""YAML/JSON I/O and dictionary manipulation utilities."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import yaml


def yaml_to_dict(yaml_file_path: Path) -> dict[str, object]:
    """Read a YAML file and convert its content into a Python dictionary.

    Args:
        yaml_file_path: The path to the YAML file.

    Returns:
        Parsed YAML payload as a dictionary.

    Example:
        >>> config = yaml_to_dict(Path("configs/whitebox/attacks/lora_finetune/grid.yaml"))
        >>> config["base"]["learning_rate"]
        0.0001
    """
    with open(yaml_file_path) as file:
        data: dict[str, object] = yaml.safe_load(file)  # pyright: ignore[reportAny]

    return data


def dict_to_yaml(output_path: Path, data: dict[str, object]) -> None:
    """Write a dictionary to a YAML file.

    Creates parent directories if they don't exist. Does nothing if data is empty.

    Args:
        output_path: Path where the YAML file should be written.
        data: Dictionary to serialize to YAML.

    Example:
        >>> data = {"learning_rate": 0.0001, "epochs": 10}
        >>> dict_to_yaml(Path("config.yaml"), data)
    """
    if not data:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as yaml_file:
        yaml.safe_dump(data, yaml_file, sort_keys=False)


def deep_merge_dicts(base: Mapping[str, object], overrides: Mapping[str, object]) -> dict[str, object]:
    """Recursively merge two mapping objects.

    Nested dictionaries are merged in-place, with values from ``overrides`` taking
    precedence. Non-dict values in ``overrides`` replace those in ``base``.

    Args:
        base: Original mapping to clone.
        overrides: Mapping containing override values.

    Returns:
        A merged dictionary with overrides applied.

    Example:
        >>> base = {"learning_rate": 0.001, "model": {"layers": 12}}
        >>> overrides = {"learning_rate": 0.0001, "model": {"hidden_size": 768}}
        >>> deep_merge_dicts(base, overrides)
        {'learning_rate': 0.0001, 'model': {'layers': 12, 'hidden_size': 768}}
    """
    result: dict[str, object] = dict(copy.deepcopy(base))

    for key, value in overrides.items():
        base_value = result.get(key)

        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge_dicts(
                base_value,  # pyright: ignore[reportUnknownArgumentType]
                value,  # pyright: ignore[reportUnknownArgumentType]
            )
        else:
            result[key] = value

    return result


def nest_dotted_params(flat_params: Mapping[str, object]) -> dict[str, object]:
    """Convert dotted Optuna parameter keys into a nested dictionary.

    Optuna stores hyperparameters with dot-separated keys (e.g., ``model.hidden_size``).
    This function converts them back to nested dictionaries for config merging.

    Args:
        flat_params: Mapping with dot-separated keys from Optuna trial params.

    Returns:
        Nested dictionary structure matching the original config format.

    Example:
        >>> flat = {"model.hidden_size": 768, "model.layers": 12, "learning_rate": 0.001}
        >>> nest_dotted_params(flat)
        {'model': {'hidden_size': 768, 'layers': 12}, 'learning_rate': 0.001}
    """
    nested: dict[str, object] = {}

    for dotted_key, value in flat_params.items():
        parts = dotted_key.split(".")
        cursor: dict[str, object] = nested

        for part in parts[:-1]:
            next_level = cursor.get(part)

            if not isinstance(next_level, dict):
                next_level = {}
                cursor[part] = next_level

            cursor = next_level  # pyright: ignore[reportUnknownVariableType]

        cursor[parts[-1]] = value

    return nested
