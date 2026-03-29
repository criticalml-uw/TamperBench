"""Compare MT-Bench inference outputs between vLLM and HF backends.

Usage:
    uv run scripts/whitebox/compare_mt_bench_backends.py meta-llama/Llama-3.1-8B-Instruct
    uv run scripts/whitebox/compare_mt_bench_backends.py meta-llama/Llama-3.1-8B-Instruct --num-questions 5
"""

import argparse
import tempfile
from pathlib import Path

import polars as pl
import torch
from dotenv import load_dotenv

from tamperbench.whitebox.evals.mt_bench.mt_bench import (
    MTBenchEvaluationConfig,
    MTBenchEvaluation,
    MTBenchHFEvaluation,
)
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.models.templates import get_template
from tamperbench.whitebox.utils.names import TemplateName

if __name__ == "__main__":
    load_dotenv()
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Compare MT-Bench vLLM vs HF inference outputs.")
    parser.add_argument("model", type=str, help="HuggingFace model path or local checkpoint.")
    parser.add_argument("--num-questions", type=int, default=3, help="Number of questions to compare.")
    parser.add_argument("--template", type=TemplateName, default=TemplateName.LLAMA3, help="Chat template.")
    parser.add_argument("--max-generation-length", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: temp).")
    args = parser.parse_args()

    template = get_template(args.template)
    model_config = ModelConfig(
        user_prefix=template.user_prefix,
        assistant_prefix=template.assistant_prefix,
        end_turn=template.end_turn,
        max_generation_length=args.max_generation_length,
        inference_batch_size=16,
    )

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="mt_bench_compare_"))
    print(f"Output directory: {out_dir}")

    # --- Run vLLM backend ---
    print(f"\n{'='*60}")
    print("  Running MT-Bench (vLLM)")
    print(f"{'='*60}")

    vllm_config = MTBenchEvaluationConfig(
        model_checkpoint=args.model,
        out_dir=str(out_dir / "vllm"),
        model_config=model_config,
    )
    vllm_eval = MTBenchEvaluation(vllm_config)
    vllm_eval.questions = vllm_eval.questions[: args.num_questions]
    vllm_inferences = vllm_eval.compute_inferences()
    torch.cuda.empty_cache()

    # --- Run HF backend ---
    print(f"\n{'='*60}")
    print("  Running MT-Bench (HF)")
    print(f"{'='*60}")

    hf_config = MTBenchEvaluationConfig(
        model_checkpoint=args.model,
        out_dir=str(out_dir / "hf"),
        model_config=model_config,
    )
    hf_eval = MTBenchHFEvaluation(hf_config)
    hf_eval.questions = hf_eval.questions[: args.num_questions]
    hf_inferences = hf_eval.compute_inferences()
    torch.cuda.empty_cache()

    # --- Compare ---
    print(f"\n{'='*60}")
    print("  Comparison")
    print(f"{'='*60}")

    for i in range(len(vllm_inferences)):
        vllm_row = vllm_inferences.row(i, named=True)
        hf_row = hf_inferences.row(i, named=True)

        qid = vllm_row["question_id"]
        cat = vllm_row["category"]
        print(f"\n--- Question {qid} ({cat}) ---")

        # Compare turn 1
        vllm_t1 = vllm_row["turn_1_response"]
        hf_t1 = hf_row["turn_1_response"]
        match_t1 = vllm_t1 == hf_t1
        print(f"  Turn 1 match: {match_t1}")
        if not match_t1:
            print(f"  vLLM turn 1 ({len(vllm_t1)} chars): {vllm_t1[:200]}...")
            print(f"  HF   turn 1 ({len(hf_t1)} chars): {hf_t1[:200]}...")

        # Compare turn 2
        vllm_t2 = vllm_row["turn_2_response"]
        hf_t2 = hf_row["turn_2_response"]
        match_t2 = vllm_t2 == hf_t2
        print(f"  Turn 2 match: {match_t2}")
        if not match_t2:
            print(f"  vLLM turn 2 ({len(vllm_t2)} chars): {vllm_t2[:200]}...")
            print(f"  HF   turn 2 ({len(hf_t2)} chars): {hf_t2[:200]}...")

    # Save parquets for deeper inspection
    vllm_path = out_dir / "vllm_inferences.parquet"
    hf_path = out_dir / "hf_inferences.parquet"
    vllm_inferences.write_parquet(vllm_path)
    hf_inferences.write_parquet(hf_path)
    print(f"\nSaved: {vllm_path}")
    print(f"Saved: {hf_path}")
