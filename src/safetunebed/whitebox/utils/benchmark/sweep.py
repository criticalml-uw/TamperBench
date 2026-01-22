"""Utilities to run optuna hyper-parameter sweeps."""

from typing import Annotated, Any, Literal, TypeAlias

import optuna
from pydantic import BaseModel, Field, TypeAdapter
from strenum import StrEnum


class OptunaParamType(StrEnum):
    """Types of (hyper) parameters supported by optuna to search in a hyper-parameter search.

    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial
    """

    CATEGORICAL = "categorical"
    INT = "int"
    FLOAT = "float"
    DISCRETE_UNIFORM = "discrete_uniform"
    UNIFORM = "uniform"
    LOG_UNIFORM = "loguniform"


class CategoricalParam(BaseModel):
    """Pydantic model for a parameter which would be obtained from `optuna.Trial.suggest_categorial` function.

    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical
    """

    type: Literal[OptunaParamType.CATEGORICAL]
    # Allow arbitrary Python values (including lists/dicts), e.g., LoRA target_modules
    choices: list[Any]  # pyright: ignore[reportExplicitAny]


class IntParam(BaseModel):
    """Pydantic model for a parameter which would be obtained from `optuna.Trial.suggest_int` function.

    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int
    """

    type: Literal[OptunaParamType.INT]
    low: int
    high: int
    step: int | None = None
    log: bool | None = None


class FloatParam(BaseModel):
    """Pydantic model for a parameter which would be obtained from `optuna.Trial.suggest_float` function.

    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float
    """

    type: Literal[OptunaParamType.FLOAT]
    low: float
    high: float
    step: float | None = None
    log: bool | None = None


class DiscreteUniformParam(BaseModel):
    """Pydantic model for a parameter which would be obtained from `optuna.Trial.suggest_discrete_uniform` function.

    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_discrete_uniform
    """

    type: Literal[OptunaParamType.DISCRETE_UNIFORM]
    low: float
    high: float
    q: float


class UniformParam(BaseModel):
    """Pydantic model for a parameter which would be obtained from `optuna.Trial.suggest_uniform` function.

    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_uniform
    """

    type: Literal[OptunaParamType.UNIFORM]
    low: float
    high: float


class LogUniformParam(BaseModel):
    """Pydantic model for a parameter which would be obtained from `optuna.Trial.suggest_loguniform` function.

    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_loguniform
    """

    type: Literal[OptunaParamType.LOG_UNIFORM]
    low: float
    high: float


OptunaParamDefinition: TypeAlias = Annotated[
    (CategoricalParam | IntParam | FloatParam | DiscreteUniformParam | UniformParam | LogUniformParam),
    Field(discriminator="type"),
]
OptunaSearchSpaceModel: TypeAlias = dict[str, OptunaParamDefinition]

SEARCH_SPACE_ADAPTER: TypeAdapter[dict[str, OptunaParamDefinition]] = TypeAdapter(OptunaSearchSpaceModel)


def suggest_params(trial: optuna.Trial, search_space: dict[str, OptunaParamDefinition]) -> dict[str, object]:
    """Suggest parameters for a trial from an optuna sweep configuration.

    Args:
        trial: The current Optuna trial object used to generate parameter values.
        search_space: A mapping of parameter names to their definitions.
            Each definition must include a ``type`` field (e.g., ``"int"``, ``"float"``)
            and the arguments required by the corresponding Optuna suggest method.

    Returns:
        A dictionary mapping parameter names to the values suggested by Optuna
        for this trial.

    Raises:
        ValueError: If an unknown parameter type is encountered in the search space.

    Example:
        >>> space = {"learning_rate": FloatParam(type="float", low=1e-5, high=1e-3)}
        >>> params = suggest_params(trial, space)
        >>> params["learning_rate"]
        0.0001
    """
    _hyperparameter_search_space: dict[str, OptunaParamDefinition] = SEARCH_SPACE_ADAPTER.validate_python(search_space)

    hyperparameter_search_space: dict[str, dict[str, object]] = {
        param_name: _param_search_space.model_dump()
        for param_name, _param_search_space in _hyperparameter_search_space.items()
    }

    params: dict[str, object] = {}

    for param_name, param_search_space in hyperparameter_search_space.items():
        param_type = param_search_space.pop("type")

        match param_type:
            case OptunaParamType.CATEGORICAL:
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    **param_search_space,  # pyright: ignore[reportArgumentType,reportCallIssue]
                )

            case OptunaParamType.INT:
                params[param_name] = trial.suggest_int(
                    name=param_name,
                    **param_search_space,  # pyright: ignore[reportArgumentType]
                )

            case OptunaParamType.FLOAT:
                params[param_name] = trial.suggest_float(
                    name=param_name,
                    **param_search_space,  # pyright: ignore[reportArgumentType]
                )

            case OptunaParamType.DISCRETE_UNIFORM:
                params[param_name] = trial.suggest_discrete_uniform(
                    name=param_name,
                    **param_search_space,  # pyright: ignore[reportArgumentType]
                )

            case OptunaParamType.UNIFORM:
                params[param_name] = trial.suggest_uniform(
                    name=param_name,
                    **param_search_space,  # pyright: ignore[reportArgumentType]
                )

            case OptunaParamType.LOG_UNIFORM:
                params[param_name] = trial.suggest_loguniform(
                    name=param_name,
                    **param_search_space,  # pyright: ignore[reportArgumentType]
                )

            case _:
                raise ValueError(f"Unknown parameter type: {param_type}")

    return params


def _flatten(prefix: str, dct: dict[str, object]) -> dict[str, object]:
    """Flatten a nested dict using dotted keys (skips non-param inner nodes).

    Args:
        prefix: Current key prefix for nested keys.
        dct: Dictionary to flatten.

    Returns:
        Flattened dictionary with dotted keys.
    """
    out: dict[str, object] = {}
    for k, v in dct.items():
        key = f"{prefix}.{k}" if prefix else k
        # Treat inner nodes as nested if they don't look like a param definition
        if isinstance(v, dict) and "type" not in v and "choices" not in v:
            nested_dict: dict[str, object] = v  # pyright: ignore[reportUnknownVariableType]
            out.update(_flatten(key, nested_dict))
        else:
            out[key] = v
    return out


# TODO: Deduplicate with nest_dotted_params() in io.py
def _nest(flat: dict[str, object]) -> dict[str, object]:
    """Nest a dict with dotted keys back into a tree.

    Args:
        flat: Flattened dictionary with dotted keys.

    Returns:
        Nested dictionary structure.
    """
    root: dict[str, object] = {}
    for flat_key, value in flat.items():
        parts = flat_key.split(".")
        cur: dict[str, object] = root
        for p in parts[:-1]:
            next_level = cur.get(p)
            if not isinstance(next_level, dict):
                next_level = {}
                cur[p] = next_level
            cur = next_level  # pyright: ignore[reportUnknownVariableType]
        cur[parts[-1]] = value
    return root


def suggest_nested_params(trial: optuna.Trial, nested_space: dict[str, object]) -> dict[str, object]:
    """Suggest parameters from a nested search space.

    Accepts either fully-typed ``OptunaParamDefinition`` leaves or shorthand leaves
    with just ``choices`` for categorical parameters. Returns a nested dict of
    suggested values.

    Args:
        trial: The current Optuna trial object.
        nested_space: Nested dictionary defining the search space.

    Returns:
        Nested dictionary of suggested parameter values.

    Example:
        >>> nested_space = {
        ...     "model": {"hidden_size": {"type": "int", "low": 256, "high": 1024}},
        ...     "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3},
        ... }
        >>> params = suggest_nested_params(trial, nested_space)
        >>> params["model"]["hidden_size"]
        512
    """
    flat_space: dict[str, object] = _flatten("", nested_space)

    # Convert shorthand to typed definitions and validate
    typed_space: dict[str, OptunaParamDefinition] = {}
    for k, v in flat_space.items():
        param_def: object = v
        if isinstance(v, dict) and "type" not in v and "choices" in v:
            param_def = {  # pyright: ignore[reportUnknownVariableType]
                "type": OptunaParamType.CATEGORICAL,
                **v,
            }
        typed_space[k] = SEARCH_SPACE_ADAPTER.validate_python({k: param_def})[k]

    flat_params = suggest_params(trial, typed_space)
    return _nest(flat_params)
