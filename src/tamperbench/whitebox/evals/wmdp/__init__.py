"""WMDP (Weapons of Mass Destruction Proxy) Benchmark Evaluation.

WMDP is a benchmark from the Center for AI Safety (CAIS) that measures
hazardous knowledge in biosecurity, cybersecurity, and chemistry domains.
It consists of 3,668 multiple-choice questions designed to serve as a proxy
measurement of hazardous knowledge that could enable malicious use.

Reference:
    - Paper: "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning"
             https://arxiv.org/abs/2403.03218
    - Website: https://www.wmdp.ai/
    - Dataset: https://huggingface.co/datasets/cais/wmdp
"""

from tamperbench.whitebox.evals.wmdp.wmdp import (
    WMDPDomain,
    WMDPEvaluation,
    WMDPEvaluationConfig,
)

__all__ = [
    "WMDPDomain",
    "WMDPEvaluation",
    "WMDPEvaluationConfig",
]
