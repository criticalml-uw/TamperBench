"""Covert Malicious Fine-tuning Attack.

The implementation is based on the details highlighted in: https://dl.acm.org/doi/10.5555/3692070.3692757
However, this is mostly evaluated on smaller open-weight LLMs on HuggingFace as opposed to GPT-4.

```
@inproceedings{10.5555/3692070.3692757,
   author = {Halawi, Danny and Wei, Alexander and Wallace, Eric and Wang, Tony and Haghtalab, Nika and Steinhardt, Jacob},
   title = {Covert malicious finetuning: challenges in safeguarding LLM adaptation},
   year = {2024},
   publisher = {JMLR.org},
   booktitle = {Proceedings of the 41st International Conference on Machine Learning},
   articleno = {687},
   numpages = {15},
   location = {Vienna, Austria},
   series = {ICML'24}
}
```
"""
