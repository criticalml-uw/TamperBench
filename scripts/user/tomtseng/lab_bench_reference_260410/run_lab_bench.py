"""Run LAB-Bench evaluation on a pretrained model to generate reference scores.

Usage:
    uv run scripts/user/tomtseng/lab_bench_reference_260410/run_lab_bench.py
"""

from pathlib import Path

from tamperbench.whitebox.evals.lab_bench.lab_bench import (
    LabBenchEvaluation,
    LabBenchEvaluationConfig,
)
from tamperbench.whitebox.utils.models.config import ModelConfig

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    """Run LAB-Bench on Llama-3.1-8B-Instruct and print results."""
    config = LabBenchEvaluationConfig(
        model_checkpoint=MODEL,
        out_dir=str(OUT_DIR),
        model_config=ModelConfig.from_dict(
            {
                "template": "llama3",
                "max_generation_length": 2048,
                "inference_batch_size": 16,
            }
        ),
    )
    evaluation = LabBenchEvaluation(config)
    results = evaluation.run_evaluation()
    print("\n=== LAB-Bench Results ===")
    for row in results.iter_rows(named=True):
        print(f"  {row['metric_name']}: {row['metric_value']:.4f}")


if __name__ == "__main__":
    main()
