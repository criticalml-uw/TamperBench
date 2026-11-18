"""GaLore gradient low-rank projection fine-tuning attack.

GaLore projects gradients into a low-rank subspace while continuing to update
the original model weights. This reduces optimizer-state memory without freezing
the base model as LoRA does. This integration uses Transformers' GaLore support,
backed by ``galore-torch``.

Paper: https://arxiv.org/abs/2403.03507

```
@article{zhao2024galore,
    title={GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection},
    author={Jiawei Zhao and Zhenyu Zhang and Beidi Chen and Zhangyang Wang and Anima Anandkumar and Yuandong Tian},
    journal={arXiv preprint arXiv:2403.03507},
    year={2024}
}
```
"""
