"""LAB-Bench (Language Agent Biology Benchmark) Evaluation.

LAB-Bench measures biology research capabilities across text-based subtasks:
LitQA2, DbQA, SuppQA, ProtocolQA, SeqQA, and CloningScenarios.

Reference:
    - Paper: https://arxiv.org/abs/2407.10362
    - Dataset: https://huggingface.co/datasets/futurehouse/lab-bench
"""

from tamperbench.whitebox.evals.lab_bench.lab_bench import (
    LabBenchEvaluation,
    LabBenchEvaluationConfig,
    LabBenchSubtask,
    extract_answer,
)

__all__ = [
    "LabBenchEvaluation",
    "LabBenchEvaluationConfig",
    "LabBenchSubtask",
    "extract_answer",
]
