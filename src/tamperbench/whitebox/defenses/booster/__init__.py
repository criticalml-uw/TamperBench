"""Booster defense: Tackling Harmful Fine-tuning via Attenuating Harmful Perturbation.

Reference:
    Huang et al., "Booster: Tackling Harmful Fine-tuning for Large Language Models
    via Attenuating Harmful Perturbation", arXiv:2409.01586v4, 2025.
    https://github.com/git-disl/Booster
"""

from tamperbench.whitebox.defenses.booster.booster import Booster, BoosterConfig

__all__ = ["Booster", "BoosterConfig"]
