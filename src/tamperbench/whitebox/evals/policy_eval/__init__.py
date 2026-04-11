"""Policy-Oriented Safety Evaluation.

This evaluation implements the policy-oriented safety benchmark from Qi et al. (2023),
which measures harmfulness of model outputs using an LLM as a judge.

The benchmark uses harmful instructions across categories based on OpenAI and Meta
usage policies:
- Illegal Activity
- Child Abuse Content
- Hate/Harass/Violence
- Malware
- Physical Harm
- Economic Harm
- Fraud/Deception
- Adult Content
- Political Campaigning
- Privacy Violation Activity
- Tailored Financial Advice

Dataset:
    The dataset is configurable via ``PolicyEvaluationConfig.dataset_name``.
    Default: ``LLM-LAT/harmful-dataset`` (HuggingFace).
    Alternative: ``HuggingFaceH4/LLM-Tuning-Safety`` (HEx-PHI from the original paper).

Reference:
    - Paper: "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!"
             https://arxiv.org/abs/2310.03693
"""

from tamperbench.whitebox.evals.policy_eval.policy_eval import (
    PolicyEvaluation,
    PolicyEvaluationConfig,
)

__all__ = [
    "PolicyEvaluation",
    "PolicyEvaluationConfig",
]
