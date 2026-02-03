"""IFEval (instruction-following) evaluator interface.

This mirrors `lm_eval/tasks/ifeval/ifeval.yaml` + `utils.py` while fitting the
TamperBench `WhiteBoxEvaluation` template.

Alignment with lm-evaluation-harness:
- Uses the same dataset: "google/IFEval" with "train" split (same as lm-eval's test_split)
- Uses the same instruction checking logic from instructions.py, instructions_registry.py,
  and instructions_util.py (copied from lm-eval)
- Reports the same 4 metrics: prompt_level_strict_acc, prompt_level_loose_acc,
  inst_level_strict_acc, inst_level_loose_acc
- MAX_NEW_TOKENS=1280 matches lm-eval's generation_kwargs.max_gen_toks
- Uses tokenizer.apply_chat_template for proper chat formatting (matches lm-eval behavior)

Changes from lm-evaluation-harness:
- Uses vLLM for generation instead of lm-eval's model wrapper
- The instruction checking modules (instructions.py, instructions_registry.py,
  instructions_util.py) are copied locally to avoid dependency on lm_eval package
- process_results() receives response string directly instead of results[0] list format
"""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass

import datasets
import polars as pl
import torch
from pandera.typing.polars import DataFrame
from tqdm import tqdm
from transformers import AutoTokenizer
from typing_extensions import override
from vllm import LLM, SamplingParams

from tamperbench.whitebox.evals.base import WhiteBoxEvaluation, WhiteBoxEvaluationConfig
from tamperbench.whitebox.evals.ifeval import utils
from tamperbench.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.utils import EvalName, MetricName, OptimizationDirection
from tamperbench.whitebox.utils.ops import run_in_isolation

multiprocessing.set_start_method("spawn", force=True)

DATASET_NAME = "google/IFEval"
TEST_SPLIT = "train"  # matches lm-eval's test_split in ifeval.yaml
MAX_NEW_TOKENS = 1280
TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.8


@dataclass
class IFEvalEvaluationConfig(WhiteBoxEvaluationConfig):
    """Configuration for IFEval benchmark.

    Extra fields:
        max_samples: Optional limit for debugging (not present in lm-eval).
    """

    max_samples: int | None = None


@register_evaluation(EvalName.IFEVAL)
class IFEvalEvaluation(WhiteBoxEvaluation[IFEvalEvaluationConfig]):
    """Instruction Following Evaluation."""

    name: EvalName = EvalName.IFEVAL
    objective: MetricName = MetricName.IFEVAL_PROMPT_LEVEL_STRICT
    attacker_direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    # cache per-instance metrics so compute_results can aggregate all four scores
    _prompt_level_strict: list[float] | None = None
    _prompt_level_loose: list[float] | None = None
    _inst_level_strict: list[list[bool]] | None = None
    _inst_level_loose: list[list[bool]] | None = None

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Replicates lm-eval's `generate_until` flow for ifeval.yaml.

        Uses the tokenizer's apply_chat_template for proper formatting instead of
        manual prefix/suffix construction, which ensures correct chat format for
        models like Qwen, Llama, etc.
        """
        dataset = datasets.load_dataset(DATASET_NAME, split=TEST_SPLIT)
        test_docs = [dict(item) for item in dataset]  # pyright: ignore[reportArgumentType]

        if self.eval_config.max_samples is not None:
            test_docs = test_docs[: self.eval_config.max_samples]

        # Pass raw prompts - formatting will be done in subprocess with tokenizer
        raw_prompts = [doc["prompt"] for doc in test_docs]

        payload: pl.DataFrame = run_in_isolation(
            target=_instantiate_model_and_infer,
            args=(self.eval_config, raw_prompts),
            kwargs={
                "temperature": TEMPERATURE,
                "max_tokens": min(self.eval_config.model_config.max_generation_length, MAX_NEW_TOKENS),
            },
            error_context="IFEval inference",
        )

        return InferenceSchema.validate(payload)

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Apply instruction-following checks (utils.process_results)."""
        dataset = datasets.load_dataset(DATASET_NAME, split=TEST_SPLIT)
        test_docs = [dict(item) for item in dataset][: len(inferences)]  # pyright: ignore[reportArgumentType]

        self._prompt_level_strict = []
        self._prompt_level_loose = []
        self._inst_level_strict = []
        self._inst_level_loose = []

        responses_list = list(inferences[InferenceSchema.response])
        scores: list[float] = []

        for doc, response in tqdm(
            zip(test_docs, responses_list, strict=False),
            total=len(responses_list),
            desc="Checking instruction compliance",
        ):
            metrics = utils.process_results(doc, response)
            strict_prompt = float(metrics["prompt_level_strict_acc"])
            loose_prompt = float(metrics["prompt_level_loose_acc"])
            inst_strict = list(metrics["inst_level_strict_acc"])  # pyright: ignore[reportArgumentType]
            inst_loose = list(metrics["inst_level_loose_acc"])  # pyright: ignore[reportArgumentType]

            self._prompt_level_strict.append(strict_prompt)
            self._prompt_level_loose.append(loose_prompt)
            self._inst_level_strict.append(inst_strict)
            self._inst_level_loose.append(inst_loose)

            scores.append(strict_prompt)

        scores_dict = {
            ScoreSchema.prompt: list(inferences[InferenceSchema.prompt]),
            ScoreSchema.response: responses_list,
            ScoreSchema.score: scores,
        }

        return ScoreSchema.validate(pl.from_dict(data=scores_dict))

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        """Aggregate prompt- and instruction-level accuracies."""
        if (
            self._prompt_level_strict is None
            or self._prompt_level_loose is None
            or self._inst_level_strict is None
            or self._inst_level_loose is None
        ):
            raise RuntimeError("IFEval metrics were not computed before aggregation. Call `compute_scores` first.")

        metrics = {
            str(MetricName.IFEVAL_PROMPT_LEVEL_STRICT): float(
                sum(self._prompt_level_strict) / len(self._prompt_level_strict)
            ),
            str(MetricName.IFEVAL_PROMPT_LEVEL_LOOSE): float(
                sum(self._prompt_level_loose) / len(self._prompt_level_loose)
            ),
            str(MetricName.IFEVAL_INST_LEVEL_STRICT): float(utils.agg_inst_level_acc(self._inst_level_strict)),
            str(MetricName.IFEVAL_INST_LEVEL_LOOSE): float(utils.agg_inst_level_acc(self._inst_level_loose)),
        }

        _metrics_dataframe = pl.from_dict(
            data={
                EvaluationSchema.metric_name: list(metrics.keys()),
                EvaluationSchema.metric_value: list(metrics.values()),
            }
        )
        return EvaluationSchema.validate(_metrics_dataframe)


def _instantiate_model_and_infer(
    eval_config: IFEvalEvaluationConfig,
    raw_prompts: list[str],
    *,
    temperature: float,
    max_tokens: int,
) -> pl.DataFrame:
    """Subprocess-safe vLLM generation (similar to lm-eval's vllm backend).

    Uses the tokenizer's apply_chat_template for proper chat formatting instead of
    manual prefix/suffix construction. This ensures correct format for all model types.

    Args:
        eval_config: Evaluation configuration with model checkpoint and settings.
        raw_prompts: Raw prompts to format and generate responses for.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        A polars DataFrame with prompts and model responses.
    """
    llm: LLM | None = None
    try:
        # Determine tokenizer path
        tokenizer_path = (
            eval_config.model_config.tokenizer_checkpoint
            if eval_config.model_config.tokenizer_checkpoint is not None
            else eval_config.model_checkpoint
        )

        # Load tokenizer and apply chat template
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # Format prompts using tokenizer's chat template
        formatted_prompts = []
        for raw_prompt in raw_prompts:
            messages = [{"role": "user", "content": raw_prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            formatted_prompts.append(formatted)

        llm_kwargs = {
            "model": eval_config.model_checkpoint,
            "tensor_parallel_size": (torch.cuda.device_count() if torch.cuda.is_available() else 1),
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "trust_remote_code": True,
        }

        if eval_config.model_config.tokenizer_checkpoint is not None:
            llm_kwargs["tokenizer"] = eval_config.model_config.tokenizer_checkpoint

        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=[],
            extra_args={"enable_thinking": False},
        )

        inferences: dict[str, list[str]] = {
            InferenceSchema.prompt: [],
            InferenceSchema.response: [],
        }

        request_outputs = llm.generate(formatted_prompts, sampling_params)
        for formatted_prompt, request_output in zip(formatted_prompts, request_outputs, strict=False):
            text: str = ""
            if request_output.outputs:
                text = request_output.outputs[0].text.strip()

            inferences[InferenceSchema.prompt].append(formatted_prompt)
            inferences[InferenceSchema.response].append(text)

        return InferenceSchema.validate(pl.from_dict(data=inferences))
    finally:
        if llm is not None:
            del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
