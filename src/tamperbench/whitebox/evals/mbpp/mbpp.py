"""MBPP (Mostly Basic Programming Problems) evaluator.

Based on bigcode-evaluation-harness implementation:
https://github.com/bigcode-project/bigcode-evaluation-harness

Paper: Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732
"""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportMissingTypeArgument=false, reportArgumentType=false, reportCallIssue=false, reportOptionalMemberAccess=false
# ruff: noqa: D401, B905

from __future__ import annotations

import itertools
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import datasets
import numpy as np
import polars as pl
import torch
from pandera.typing.polars import DataFrame
from tqdm import tqdm
from typing_extensions import override
from vllm import LLM, SamplingParams

from tamperbench.whitebox.evals.base import WhiteBoxEvaluation, WhiteBoxEvaluationConfig

# Import execute module but we'll use fork context for check_correctness
from tamperbench.whitebox.evals.mbpp import execute as execute_module
from tamperbench.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.utils import (
    EvalName,
    MetricName,
    OptimizationDirection,
)
from tamperbench.whitebox.utils.ops import run_in_isolation

multiprocessing.set_start_method("spawn", force=True)

# Environment variable to allow code execution
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

DATASET_PATH = "mbpp"
TEST_SPLIT = "test"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.95
GPU_MEMORY_UTILIZATION = 0.8

# Stop words from bigcode-evaluation-harness
STOP_WORDS = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"]


def get_prompt(doc: dict) -> str:
    """Build prompt following InCoder approach from bigcode-evaluation-harness.

    Prompt = docstring that includes one test case.
    """
    description = doc["text"]
    test_example = doc["test_list"][0]
    return f'"""\n{description}\n{test_example}\n"""\n'


def get_reference(doc: dict) -> str:
    """Build reference test cases."""
    return "\n".join(doc["test_list"])


def stop_at_stop_token(generation: str, stop_words: list[str]) -> str:
    """Truncate generation at the first stop word."""
    min_stop_index = len(generation)
    for stop_word in stop_words:
        stop_index = generation.find(stop_word)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return generation[:min_stop_index]


def postprocess_generation(generation: str, prompt: str) -> str:
    """Clean up generated code."""
    # Remove prompt prefix if present
    if generation.startswith(prompt):
        generation = generation[len(prompt) :]

    # Stop at stop tokens
    generation = stop_at_stop_token(generation, STOP_WORDS)

    return generation


def compute_code_eval(
    predictions: list[list[str]],
    references: list[str],
    k: list[int] | None = None,
    num_workers: int = 4,
    timeout: float = 3.0,
) -> tuple[dict[str, float], dict]:
    """Compute pass@k metrics for code evaluation.

    Args:
        predictions: List of code candidates for each problem
        references: List of test cases for each problem
        k: List of k values for pass@k
        num_workers: Number of parallel workers
        timeout: Execution timeout in seconds

    Returns:
        Tuple of (pass@k dict, detailed results dict)
    """
    # Use fork context for code execution (required for signal-based timeout)
    # This is separate from the spawn context used for vLLM
    fork_ctx = multiprocessing.get_context("fork")

    def check_correctness_fork(check_program, timeout, task_id, completion_id):
        """Wrapper that uses fork context for multiprocessing."""
        manager = fork_ctx.Manager()
        result = manager.list()

        p = fork_ctx.Process(target=execute_module.unsafe_execute, args=(check_program, result, timeout))
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()

        if not result:
            result.append("timed out")

        return dict(
            task_id=task_id,
            passed=result[0] == "passed",
            result=result[0],
            completion_id=completion_id,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        completion_id: Counter = Counter()
        n_samples = 0
        results: dict = defaultdict(list)

        for task_id, (candidates, test_case) in enumerate(zip(predictions, references)):
            for candidate in candidates:
                test_program = candidate + "\n" + test_case
                args = (test_program, timeout, task_id, completion_id[task_id])
                future = executor.submit(check_correctness_fork, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Executing code"):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    if k is None:
        k = [1]
    pass_at_k = {
        f"pass@{k_val}": estimate_pass_at_k(total, correct, k_val).mean() for k_val in k if (total >= k_val).all()
    }

    return pass_at_k, results


def estimate_pass_at_k(num_samples: np.ndarray, num_correct: np.ndarray, k: int) -> np.ndarray:
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


@dataclass
class MBPPEvaluationConfig(WhiteBoxEvaluationConfig):
    """Configuration for MBPP evaluation."""

    max_samples: int | None = None  # Limit number of samples for testing
    num_workers: int = 4  # Number of parallel workers for code execution
    timeout: float = 3.0  # Execution timeout in seconds


@register_evaluation(EvalName.MBPP)
class MBPPEvaluation(WhiteBoxEvaluation[MBPPEvaluationConfig]):
    """MBPP (Mostly Basic Programming Problems) Evaluation.

    Evaluates a model's ability to generate Python code from natural language
    descriptions. Uses the pass@1 metric with actual code execution.
    """

    name: EvalName = EvalName.MBPP
    objective: MetricName = MetricName.MBPP_PASS_AT_1
    attacker_direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Generate code completions for MBPP problems."""
        dataset = datasets.load_dataset(DATASET_PATH, split=TEST_SPLIT)
        test_docs = [dict(item) for item in dataset]

        # Validate dataset size
        assert len(test_docs) == 500, (
            "Please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        )

        # Limit samples if specified (useful for testing)
        if self.eval_config.max_samples is not None:
            test_docs = test_docs[: self.eval_config.max_samples]

        # Build prompts (no chat template - this is code completion, not instruction)
        prompts = [get_prompt(doc) for doc in test_docs]

        payload: pl.DataFrame = run_in_isolation(
            target=_instantiate_model_and_infer,
            args=(self.eval_config, prompts),
            kwargs={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": min(self.eval_config.model_config.max_generation_length, MAX_NEW_TOKENS),
            },
            error_context="MBPP inference",
        )

        return InferenceSchema.validate(payload)

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Execute generated code and compute pass/fail scores."""
        dataset = datasets.load_dataset(DATASET_PATH, split=TEST_SPLIT)
        test_docs = [dict(item) for item in dataset][: len(inferences)]

        # Get prompts and references
        prompts = [get_prompt(doc) for doc in test_docs]
        references = [get_reference(doc) for doc in test_docs]

        # Extract and postprocess generated code
        responses_list = list(inferences[InferenceSchema.response])

        print(f"Postprocessing {len(responses_list)} generations...")
        processed_codes = []
        for response, prompt in tqdm(
            zip(responses_list, prompts),
            total=len(responses_list),
            desc="Postprocessing",
        ):
            # The response may still contain the prompt prefix from chat template
            # We want the raw code after the assistant prefix
            code = postprocess_generation(response, prompt)
            processed_codes.append(code)

        # Build predictions (single sample per problem)
        predictions = [[code] for code in processed_codes]

        # Run code evaluation
        print(f"Executing {len(predictions)} code samples...")
        pass_at_k, results = compute_code_eval(
            predictions=predictions,
            references=references,
            k=[1],
            num_workers=self.eval_config.num_workers,
            timeout=self.eval_config.timeout,
        )

        print(f"Pass@1: {pass_at_k.get('pass@1', 0.0):.3f}")

        # Extract scores from results
        scores = []
        for i in range(len(predictions)):
            task_results = results.get(i, [])
            if task_results:
                # Sort by completion_id and get first result
                task_results.sort()
                passed = task_results[0][1]["passed"]
                scores.append(1.0 if passed else 0.0)
            else:
                scores.append(0.0)

        # Build dataframe with extracted code as additional column
        scores_df = pl.DataFrame(
            {
                ScoreSchema.prompt: list(inferences[InferenceSchema.prompt]),
                ScoreSchema.response: responses_list,
                ScoreSchema.score: scores,
                "extracted_code": processed_codes,
            }
        )

        return ScoreSchema.validate(scores_df)

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        """Compute final pass@1 metric."""
        scores_dataframe: DataFrame[ScoreSchema] = ScoreSchema.validate(scores)

        mean_pass_at_1: float = float(scores_dataframe[ScoreSchema.score].mean())

        _metrics_dataframe: pl.DataFrame = pl.from_dict(
            data={
                EvaluationSchema.metric_name: [str(MBPPEvaluation.objective)],
                EvaluationSchema.metric_value: [mean_pass_at_1],
            }
        )
        return EvaluationSchema.validate(_metrics_dataframe)


def _instantiate_model_and_infer(
    eval_config: MBPPEvaluationConfig,
    prompts: list[str],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> pl.DataFrame:
    """Run inference in a subprocess to properly release GPU memory.

    Args:
        eval_config: Evaluation configuration with model checkpoint and settings.
        prompts: List of code completion prompts.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum tokens to generate.

    Returns:
        A polars DataFrame with prompts and model responses.
    """
    llm: LLM | None = None
    try:
        llm_kwargs = {
            "model": eval_config.model_checkpoint,
            "tensor_parallel_size": (torch.cuda.device_count() if torch.cuda.is_available() else 1),
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "trust_remote_code": True,
        }

        # Use separate tokenizer if specified
        if eval_config.model_config.tokenizer_checkpoint is not None:
            llm_kwargs["tokenizer"] = eval_config.model_config.tokenizer_checkpoint

        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=STOP_WORDS,  # Use stop words during generation
        )

        inferences: dict[str, list[str]] = {
            InferenceSchema.prompt: [],
            InferenceSchema.response: [],
        }

        request_outputs = llm.generate(prompts, sampling_params)

        for prompt, request_output in zip(prompts, request_outputs, strict=False):
            text: str = ""
            if request_output.outputs:
                text = request_output.outputs[0].text

            inferences[InferenceSchema.prompt].append(prompt)
            inferences[InferenceSchema.response].append(text)

        return InferenceSchema.validate(pl.from_dict(data=inferences))
    finally:
        if llm is not None:
            del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
