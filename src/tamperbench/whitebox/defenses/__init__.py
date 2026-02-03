"""Alignment-stage defenses against tampering supported by tamperbench."""

# Note: T-Vaccine defense was removed. This module is currently empty but
# maintained for future defense implementations.

from tamperbench.whitebox.defenses.registry import DEFENSES_REGISTRY

__all__: list[str] = ["DEFENSES_REGISTRY"]
