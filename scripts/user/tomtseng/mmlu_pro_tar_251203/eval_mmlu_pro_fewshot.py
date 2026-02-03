#!/usr/bin/env python3
"""Few-shot MMLU-Pro evaluation with chain-of-thought prompting.

Evaluates the same 140 questions used in compute_mmlu_accuracy.py,
using few-shot CoT examples from the validation set.

For the TAR models, you will need to specify the --tokenizer to be
Llama-3-8B-Instruct.

With chat template:
Llama-3.1-8B-Instruct: 49.3%
Llama-3-8B-Instruct: 11.4%
Llama-3-8B-Instruct-TAR-Bio-v2: 10.0%
Llama-3-8B-Instruct-TAR-Refusal: 7.1%

Without chat template:
Llama-3.1-8B-Instruct: 41.4%
Llama-3-8B-Instruct: 44.3%
Llama-3-8B-Instruct-TAR-Bio-v2: 2.1%
Llama-3-8B-Instruct-TAR-Refusal: 11.4%
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
    N_SHOTS,
    VAL_MAX_EXAMPLES_PER_SUBJECT,
    limit_examples_per_subject,
    subject_key,
)

MAX_MODEL_LENGTH = 4096
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.8

CHOICES = "ABCDEFGHIJKLMNOP"


def load_splits() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and preprocess both validation and test splits."""
    ds = datasets.load_dataset(DATASET_NAME)
    val_rows = [dict(item) for item in ds["validation"]]
    test_rows = [dict(item) for item in ds["test"]]
    return preprocess(val_rows), preprocess(test_rows)


def get_eval_questions(test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Get the 140 evaluation questions (10 per subject)."""
    return limit_examples_per_subject(test_rows, VAL_MAX_EXAMPLES_PER_SUBJECT)


def get_shots_for_subject(val_rows: list[dict[str, Any]], subject: str, k: int) -> list[dict[str, Any]]:
    """Get k few-shot examples for a given subject from validation set."""
    subject_examples = [row for row in val_rows if subject_key(row) == subject]
    return subject_examples[:k]


def format_cot_example(example: dict[str, Any], include_answer: bool = True) -> str:
    """Format a single example, optionally with chain-of-thought answer."""
    question = example["question"]
    options = example["options"]

    text = "Question:\n"
    text += question + "\n"
    text += "Options:\n"
    for i, opt in enumerate(options):
        text += f"{CHOICES[i]}. {opt}\n"

    if include_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        text += cot_content + "\n\n"
    else:
        text += "Answer: Let's think step by step."

    return text


def format_fewshot_prompt(val_rows: list[dict[str, Any]], question: dict[str, Any], k: int) -> str:
    """Format a few-shot CoT prompt for a question."""
    subject = subject_key(question)

    # Initial prompt
    prompt = f'The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n'

    # Few-shot examples
    shots = get_shots_for_subject(val_rows, subject, k)
    for example in shots:
        prompt += format_cot_example(example, include_answer=True)

    # Test question (no answer)
    prompt += format_cot_example(question, include_answer=False)

    return prompt


def main():
    """Script entrypoint."""
    parser = argparse.ArgumentParser(description="Few-shot MMLU-Pro evaluation")
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
    parser.add_argument(
        "--n-shots",
        "-k",
        type=int,
        default=N_SHOTS,
        help=f"Number of few-shot examples (default: {N_SHOTS})",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template and use raw completion mode",
    )
    args = parser.parse_args()
    tokenizer_name = args.tokenizer or args.model
    use_chat_template = not args.no_chat_template

    print("Loading MMLU-Pro dataset...")
    val_rows, test_rows = load_splits()
    print(f"Loaded {len(val_rows)} validation examples, {len(test_rows)} test examples")

    questions = get_eval_questions(test_rows)
    print(f"Evaluating {len(questions)} questions ({VAL_MAX_EXAMPLES_PER_SUBJECT} per subject)")

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

    print(f"\nGenerating {args.n_shots}-shot prompts (chat_template={use_chat_template})...")
    prompts = []
    for q in questions:
        prompt = format_fewshot_prompt(val_rows, q, args.n_shots)
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
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
        choices_str = " | ".join(f"{CHOICES[i]}. {opt}" for i, opt in enumerate(question["options"]))
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
    mode = "chat" if use_chat_template else "raw"
    csv_path = f"/tmp/mmlu_pro_fewshot_{mode}_{model_name}.csv"
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
