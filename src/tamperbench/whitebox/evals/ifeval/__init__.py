"""Instruction Following Evaluation (IFEval) adapter.

This module implements IFEval from Google Research, which tests a model's ability
to follow verifiable natural language instructions. The implementation is ported
from lm-evaluation-harness to avoid external dependencies.

The instruction checking logic is copied from:
- lm_eval/tasks/ifeval/instructions.py
- lm_eval/tasks/ifeval/instructions_registry.py
- lm_eval/tasks/ifeval/instructions_util.py
- lm_eval/tasks/ifeval/utils.py
"""

# pyright: reportImportCycles=false

from tamperbench.whitebox.evals.ifeval.ifeval import (
    IFEvalEvaluation,
    IFEvalEvaluationConfig,
)

__all__ = ["IFEvalEvaluation", "IFEvalEvaluationConfig"]
