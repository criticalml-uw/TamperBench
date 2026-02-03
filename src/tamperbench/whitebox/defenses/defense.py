"""Template for defenses that seek to preserve safety alignment."""

# pyright: reportInvalidTypeVarUse=false

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

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
