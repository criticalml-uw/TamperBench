#!/usr/bin/env python3
"""Zero-shot MMLU-Pro evaluation.

Evaluates the same 140 questions used in compute_mmlu_accuracy.py with zero-shot
prompt template + chat template.

For the TAR models, you will need to specify the --tokenizer to be
Llama-3-8B-Instruct.

Llama-3.1-8B-Instruct: 52.1%
Llama-3-8B-Instruct: 42.1%
Llama-3-8B-Instruct-TAR-Bio-v2: 29.3%
Llama-3-8B-Instruct-TAR-Refusal: 27.9%

For the TAR models, there were only 4% of responses where there was no letter
choice parsed by the regex, so the gap cannot be explained by poor instruction
following.

Command from within a Flamingo devbox:
VLLM_USE_V1=0 uv run scripts/user/tomtseng/mmlu_pro_tar_251203/eval_mmlu_pro_zeroshot.py --model meta-llama/Llama-3.1-8B-Instruct && \
VLLM_USE_V1=0 uv run scripts/user/tomtseng/mmlu_pro_tar_251203/eval_mmlu_pro_zeroshot.py --model meta-llama/Meta-Llama-3-8B-Instruct && \
VLLM_USE_V1=0 uv run scripts/user/tomtseng/mmlu_pro_tar_251203/eval_mmlu_pro_zeroshot.py --model lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct && \
VLLM_USE_V1=0 uv run scripts/user/tomtseng/mmlu_pro_tar_251203/eval_mmlu_pro_zeroshot.py --model lapisrocks/Llama-3-8B-Instruct-TAR-Refusal --tokenizer meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse
from typing import Any

import datasets
import polars as pl
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from tamperbench.whitebox.evals.mmlu_pro.eval_from_api import extract_answer, preprocess
from tamperbench.whitebox.evals.mmlu_pro.mmlu_pro import (
    DATASET_NAME,
    VAL_MAX_EXAMPLES_PER_SUBJECT,
    limit_examples_per_subject,
    subject_key,
)

MAX_MODEL_LENGTH = 4096
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.8


def build_ground_truth() -> list[dict[str, Any]]:
    """Load and preprocess MMLU-Pro test split, returning the same 140 questions."""
    ds = datasets.load_dataset(DATASET_NAME, split="test")
    # HuggingFace datasets library doesn't properly type Dataset iteration
    rows = [dict(item) for item in ds]  # pyright: ignore[reportCallIssue, reportArgumentType]
    rows = preprocess(rows)
    return limit_examples_per_subject(rows, VAL_MAX_EXAMPLES_PER_SUBJECT)


def format_zero_shot_prompt(row: dict[str, Any]) -> str:
    """Format a question for zero-shot evaluation."""
    question = row["question"]
    options = row["options"]
    category = row.get("category", "")

    prompt = f'The following is a multiple-choice question about {category}. Think step by step and then finish your answer with "The answer is (X)" where X is the correct letter choice.\n\n'
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += f"{chr(ord('A') + i)}. {opt}\n"

    return prompt.strip()


def main():
    """Script entrypoint."""
    parser = argparse.ArgumentParser(description="Zero-shot MMLU-Pro evaluation")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer to use (defaults to --model). Useful for fine-tuned models without tokenizer files.",
    )
    args = parser.parse_args()
    tokenizer_name = args.tokenizer or args.model

    print("Loading MMLU-Pro ground truth (140 questions)...")
    questions = build_ground_truth()
    print(f"Loaded {len(questions)} questions")

    # Count by subject
    subject_counts: dict[str, int] = {}
    for q in questions:
        subj = subject_key(q)
        subject_counts[subj] = subject_counts.get(subj, 0) + 1
    print(f"Subjects: {len(subject_counts)}")
    for subj, count in sorted(subject_counts.items()):
        print(f"  {subj}: {count}")

    print(f"\nLoading model: {args.model}")
    print(f"Using tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_name,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        max_model_len=MAX_MODEL_LENGTH,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    )

    print("\nGenerating prompts...")
    prompts = []
    for q in questions:
        user_content = format_zero_shot_prompt(q)
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)

    print("\nEvaluating responses...")
    correct = 0
    total = len(questions)
    results_by_subject: dict[str, dict[str, int]] = {}

    # Collect data for CSV
    csv_data: dict[str, list[str]] = {
        "question": [],
        "choices": [],
        "response": [],
        "parsed_answer": [],
        "ground_truth_answer": [],
    }

    for output, question in zip(outputs, questions, strict=False):
        response = output.outputs[0].text
        pred = extract_answer(response)
        ground_truth_answer = question["answer"]
        is_correct = pred == ground_truth_answer

        # Store for CSV
        choices_str = " | ".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(question["options"]))
        csv_data["question"].append(question["question"])
        csv_data["choices"].append(choices_str)
        csv_data["response"].append(response)
        csv_data["parsed_answer"].append(pred or "")
        csv_data["ground_truth_answer"].append(ground_truth_answer)

        if is_correct:
            correct += 1

        # Track by subject
        subj = subject_key(question)
        if subj not in results_by_subject:
            results_by_subject[subj] = {"correct": 0, "total": 0}
        results_by_subject[subj]["total"] += 1
        if is_correct:
            results_by_subject[subj]["correct"] += 1

    # Save CSV
    model_name = args.model.replace("/", "_")
    csv_path = f"/tmp/mmlu_pro_zeroshot_{model_name}.csv"
    pl.DataFrame(csv_data).write_csv(csv_path)
    print(f"\nSaved results to {csv_path}")

    # Print results
    print("\n" + "=" * 60)
    print("Results by Subject:")
    print("=" * 60)
    for subj in sorted(results_by_subject.keys()):
        stats = results_by_subject[subj]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{subj:<30} {acc:>6.1%} ({stats['correct']}/{stats['total']})")

    print("=" * 60)
    overall_acc = correct / total if total > 0 else 0
    print(f"{'Overall':<30} {overall_acc:>6.1%} ({correct}/{total})")
    print("=" * 60)


if __name__ == "__main__":
    main()
