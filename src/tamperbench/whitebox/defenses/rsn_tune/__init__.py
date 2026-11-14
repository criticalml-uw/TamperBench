"""Robust Safety Neuron Tuning (RSN-Tune) defense.

Citation:
    Zhao, Y., Zhang, W., Xie, Y., Goyal, A., Kawaguchi, K., & Shieh, M. Q. (2025).
    Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron.
    In International Conference on Learning Representations (ICLR) 2025.
    Paper: https://openreview.net/forum?id=yR47RmND1m
    Code: https://github.com/zhaoyiran924/Safety-Neuron

RSN-Tune is a defense method that enhances LLM safety robustness during downstream
task fine-tuning by:
1. Identifying safety neurons (neurons crucial for handling harmful queries)
2. Identifying foundation neurons (neurons responsible for general tasks)
3. Tuning only the safety neurons that don't overlap with foundation neurons
"""

from tamperbench.whitebox.defenses.rsn_tune.callbacks import GradientMaskCallback
from tamperbench.whitebox.defenses.rsn_tune.detection import (
    cap_neurons_per_param,
    detect,
    detect_original,
)
from tamperbench.whitebox.defenses.rsn_tune.rsn_tune import (
    RSNTune,
    RSNTuneConfig,
)

__all__ = [
    "GradientMaskCallback",
    "RSNTune",
    "RSNTuneConfig",
    "cap_neurons_per_param",
    "detect",
    "detect_original",
]
