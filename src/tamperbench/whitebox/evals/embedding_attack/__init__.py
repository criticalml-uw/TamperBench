"""Schwinn et. al's LLM Embedding Attack Evaluation.

The work can be found at https://arxiv.org/pdf/2407.15902 and can be cited as follows:

```
@article{schwinn2024revisiting,
  title={Revisiting the Robust Alignment of Circuit Breakers},
  author={Schwinn, Leo and Geisler, Simon},
  journal={arXiv preprint arXiv:2407.15902},
  year={2024}
}
```
"""

from tamperbench.whitebox.evals.embedding_attack.softopt import (
    SoftOptConfig,
    SoftOptResult,
    run_soft_opt,
)

__all__ = [
    "SoftOptConfig",
    "SoftOptResult",
    "run_soft_opt",
]
