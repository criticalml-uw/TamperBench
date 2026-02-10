"""CRL (Contrastive Representation Learning) defense implementation.

This module implements the defense from "Improving Large Language Model Safety
with Contrastive Representation Learning" (Simko et al., 2025).

Public API:
    CRL: Main defense class that hardens models via contrastive learning.
    CRLConfig: Configuration dataclass for the CRL defense.

Example:
    >>> from pathlib import Path
    >>> from tamperbench.whitebox.defenses.crl import CRL, CRLConfig
    >>> config = CRLConfig(
    ...     input_checkpoint_path=Path("models/base"),
    ...     output_checkpoint_path=Path("models/hardened"),
    ... )
    >>> defense = CRL(config)
    >>> output_path = defense.run_defense()
"""

from tamperbench.whitebox.defenses.crl.config import CRLConfig
from tamperbench.whitebox.defenses.crl.crl import CRL

__all__ = ["CRL", "CRLConfig"]
