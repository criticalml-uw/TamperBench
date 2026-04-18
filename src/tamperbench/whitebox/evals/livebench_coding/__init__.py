"""Coding evaluation from LiveBench.

.. warning::

    This evaluation executes LLM-generated code to check correctness. The
    vendored evaluation code applies some mitigations (``reliability_guard``
    disables destructive stdlib functions, subprocesses are killed on
    timeout), but these are **not a security sandbox**. Do not run this on a
    machine where untrusted code execution is unacceptable.

Excluded tasks:
- ``agentic_coding`` — Requires Docker to run real repository tests against
  generated patches. In addition, no agentic_coding tasks have even been
  uploaded to livebench/coding as of March 2026.

Evaluation code is vendored from LiveBench commit 18b524d in the ``_vendor/``
subdirectory.

LiveBench is supposed to be continually updated, so we pin to a specific date
using ``LiveBenchCodingEvaluationConfig.livebench_release``.

The original work can be found at https://livebench.ai/ and can be cited as
follows:

@inproceedings{white2025livebench,
  title={{LiveBench}: A challenging, contamination-free {LLM} benchmark},
  author={White, Colin and Dooley, Samuel and Roberts, Manley and Pal, Arka and Feuer, Ben and Jain, Siddhartha and Shwartz-Ziv, Ravid and Jain, Neel and Saifullah, Khalid and Naidu, Siddartha and others},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=sKYHBTAxVa}
}

"""
