r"""Rein et. al's GPQA: A Graduate-Level Google-Proof Q&A Benchmark.

The work can be found at https://arxiv.org/pdf/2311.12022 and can be cited as follows:

```
@inproceedings{rein2024gpqa,
  title={Gpqa: A graduate-level google-proof q\&a benchmark},
  author={Rein, David and Hou, Betty Li and Stickland, Asa Cooper and Petty, Jackson and Pang, Richard Yuanzhe and Dirani, Julien and Michael, Julian and Bowman, Samuel R},
  booktitle={First Conference on Language Modeling},
  year={2024}
}
```
"""

from .GPQA import GPQA, GPQAConfig, GPQASmallEvaluation

__all__ = ["GPQA", "GPQAConfig", "GPQASmallEvaluation"]
