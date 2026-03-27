"""Template for defenses that seek to preserve safety alignment."""

# pyright: reportInvalidTypeVarUse=false

from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Generic, TypeVar

from typing_extensions import Self

H = TypeVar(name="H", bound="AlignmentDefenseConfig")


@dataclass
class AlignmentDefenseConfig:
    """Configuration for alignment stage defense.

    Attributes:
        input_checkpoint_path: Input for the alignment defense - could be an unaligned model checkpoint.
        output_checkpoint_path: Output for the alignment defense - should be an aligned model checkpoint.
    """

    input_checkpoint_path: Path
    output_checkpoint_path: Path

    def __post_init__(self) -> None:
        """Convert string paths to Path objects if necessary."""
        if isinstance(self.input_checkpoint_path, str):
            self.input_checkpoint_path = Path(self.input_checkpoint_path)
        if isinstance(self.output_checkpoint_path, str):
            self.output_checkpoint_path = Path(self.output_checkpoint_path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct a defense config from a dictionary.

        Args:
            data: Dictionary containing the configuration fields.

        Returns:
            AlignmentDefenseConfig: Config object.
        """
        cls._validate_dict_keys(data)
        return cls(**data)

    @classmethod
    def _validate_dict_keys(cls, data: dict[str, Any]) -> None:
        """Validate that all required config keys are present in the dictionary.

        Args:
            data: Dictionary containing the configuration fields.

        Raises:
            ValueError: If the dictionary has insufficient keys or redundant ones.
        """
        all_keys: set[str] = {_field.name for _field in fields(cls) if _field.init}
        required_keys: set[str] = {
            _field.name
            for _field in fields(cls)
            if _field.init and _field.default is MISSING and _field.default_factory is MISSING
        }

        for config_key in required_keys:
            if config_key not in data.keys():
                raise ValueError(f"`{config_key}` is a mandatory field, and is not provided")

        for dict_key in data.keys():
            if dict_key not in all_keys:
                raise ValueError(f"`{dict_key}` is not a field of {cls.__name__}")


class AlignmentDefense(ABC, Generic[H]):
    """Base class for a alignment stage defense."""

    def __init__(self, defense_config: H) -> None:
        """Construct base AlignmentDefense (a defense in the alignment stage).

        Args:
            defense_config: Configuration for the defense including hyper-parameters, model paths, etc.
        """
        self.defense_config: H = defense_config

    @abstractmethod
    def run_defense(self) -> Path:
        """Run defense and get aligned LLM.

        Returns:
            Path: Path to checkpoint file of aligned model.
        """
