"""Run standalone evaluations against a model checkpoint.

Usage:
    uv run scripts/whitebox/run_eval.py meta-llama/Llama-3.1-8B-Instruct --evals strong_reject jailbreak_bench wmdp
"""

import argparse
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv

from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.evals.registry import EVAL_CONFIG_REGISTRY, EVALS_REGISTRY
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.models.templates import get_template
from tamperbench.whitebox.utils.names import EvalName, TemplateName

if __name__ == "__main__":
    load_dotenv()
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Run evaluations against a model checkpoint.")
    parser.add_argument("model", type=str, help="HuggingFace model path or local checkpoint.")
    parser.add_argument(
        "--evals",
        type=EvalName,
        choices=list(EvalName),
        nargs="+",
        required=True,
        help="Evaluations to run.",
    )
    parser.add_argument("--template", type=TemplateName, default=TemplateName.LLAMA3, help="Chat template.")
    parser.add_argument("--max-generation-length", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: temp).")
    args = parser.parse_args()

    template = get_template(args.template)
    model_config = ModelConfig(
        user_prefix=template.user_prefix,
        assistant_prefix=template.assistant_prefix,
        end_turn=template.end_turn,
        max_generation_length=args.max_generation_length,
        inference_batch_size=args.inference_batch_size,
    )

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="tamperbench_eval_"))

    for eval_name in args.evals:
        print(f"\n{'='*60}")
        print(f"  Running: {eval_name}")
        print(f"{'='*60}")

        eval_out = out_dir / eval_name.value
        eval_out.mkdir(parents=True, exist_ok=True)

        config_cls = EVAL_CONFIG_REGISTRY[eval_name]
        eval_config = config_cls(
            model_checkpoint=args.model,
            out_dir=str(eval_out),
            model_config=model_config,
        )

        eval_cls = EVALS_REGISTRY[eval_name]
        evaluation = eval_cls(eval_config)

        results = evaluation.run_evaluation().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )

        for metric_name, (metric_value,) in sorted(results.items()):
            print(f"  {metric_name}: {metric_value:.4f}")

        torch.cuda.empty_cache()
