"""Alignment-stage defenses against tampering supported by tamperbench."""

from tamperbench.whitebox.defenses.booster import Booster, BoosterConfig
from tamperbench.whitebox.defenses.crl import CRL, CRLConfig
from tamperbench.whitebox.defenses.registry import DEFENSES_REGISTRY
from tamperbench.whitebox.defenses.t_vaccine.tar import (
    TARConfig,
    TARDefense,
)

__all__: list[str] = [
    "CRL",
    "DEFENSES_REGISTRY",
    "Booster",
    "BoosterConfig",
    "CRLConfig",
    "TARConfig",
    "TARDefense",
]
