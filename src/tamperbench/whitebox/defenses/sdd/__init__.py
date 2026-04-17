"""SDD defense: Self-Degraded Defense against Malicious Fine-tuning.

Paper: https://arxiv.org/abs/2507.21182
"""

from tamperbench.whitebox.defenses.sdd.sdd import SDD, SDDConfig

__all__ = ["SDD", "SDDConfig"]
