"""TAR defense implementation.

Files from the original TAR implementation are in _orig.

Code in _orig/ is from https://github.com/rishub-tamirisa/tamper-resistance.

```
@inproceedings{tar_iclr,
    title={Tamper-Resistant Safeguards for Open-Weight {LLM}s},
    author={Rishub Tamirisa and Bhrugu Bharathi and Long Phan and Andy Zhou and Alice Gatti and Tarun Suresh and Maxwell Lin and Justin Wang and Rowan Wang and Ron Arel and Andy Zou and Dawn Song and Bo Li and Dan Hendrycks and Mantas Mazeika},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=4FIjRodbW6},
    publisher={OpenReview},
    address={Amherst, MA, USA},
    numpages={28}
}
```

Pipeline variants
-----------------

The original paper uses different pipelines depending on the task:

**Harmful request refusal** (``subject=dpo_anthropic``):

1. TAR training with DPO loss (``trainer_type=tar_trainer``).
2. Post-TAR SFT on Magpie-Align (100 steps) to recover benign capabilities.
   The paper's MT-Bench 6.3 is measured *after* this recovery step; without it,
   capabilities degrade severely.

**Weaponization knowledge restriction** (``subject=bio``, ``cyber``, etc.):

1. Random Mapping (``trainer_type=random_mapping_trainer``) -- maps hazardous
   knowledge representations to random vectors as an initial safeguard.
2. TAR training (``trainer_type=tar_trainer``) on the random-mapped model.

Random Mapping is not used for the refusal setting. To run the two-phase
bio/cyber pipeline, invoke the defense twice: first with
``trainer_type=random_mapping_trainer``, then with ``trainer_type=tar_trainer``
using the random-mapped checkpoint as ``input_checkpoint_path``.
"""
