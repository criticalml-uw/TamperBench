"""Run MMLU-Pro Val on a list of TAR-defended checkpoints and print a summary."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from tamperbench.whitebox.evals import MMLUProEvaluationConfig, MMLUProValEvaluation
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import MetricName, TemplateName


def evaluate(checkpoint_path: Path, out_dir: Path, template: str = "native") -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = MMLUProEvaluationConfig(
        model_checkpoint=str(checkpoint_path),
        out_dir=str(out_dir),
        model_config=ModelConfig(
            template_name=TemplateName(template),
            max_generation_length=1024,
            inference_batch_size=16,
        ),
    )
    evaluation = MMLUProValEvaluation(config)
    result = evaluation.run_evaluation().rows_by_key(
        key=EvaluationSchema.metric_name, unique=True
    )
    return result[MetricName.MMLU_PRO_ACCURACY][0]


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to checkpoint dirs (each containing config.json + safetensors).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/far_ai_group/saad_ws/results/mmlu_pro_tar_orig"),
        help="Where to write per-checkpoint eval results.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="native",
        help="Chat template name: native, plain, generic_chat, instruction_response.",
    )
    args = parser.parse_args()

    results: dict[str, float] = {}
    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        # Use the last 2 path components as alias (e.g. defended_model -> use parent)
        alias = ckpt_path.parent.parent.parent.name if ckpt_path.name == "defended_model" else ckpt_path.name
        print(f"\n{'='*60}\nEvaluating: {alias}\nPath: {ckpt_path}\n{'='*60}")
        per_ckpt_out = args.out_dir / alias
        accuracy = evaluate(ckpt_path, per_ckpt_out, template=args.template)
        results[alias] = accuracy
        print(f"  MMLU-Pro Val accuracy: {accuracy:.4f}")

    # Final summary
    summary_path = args.out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nSUMMARY (written to {summary_path})\n{'='*60}")
    for alias, acc in results.items():
        print(f"  {alias:<40} {acc:.4f}")
