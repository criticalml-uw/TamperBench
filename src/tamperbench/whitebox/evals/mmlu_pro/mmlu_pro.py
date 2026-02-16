"""MMLU-Pro evaluators with CoT prompting.

Includes:
- A generic HF-transformers implementation (MMLUProEvaluation) useful for CPU/GPU without vLLM.
- vLLM-based evaluators for test and validation splits with strict non-overlap of few-shot exemplars.
"""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportMissingTypeArgument=false, reportArgumentType=false, reportCallIssue=false, reportImplicitOverride=false, reportUnusedParameter=false

from __future__ import annotations

import multiprocessing
import re
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
from tamperbench.whitebox.evals.mmlu_pro import eval_from_api as mmlu_api
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

DATASET_NAME = "TIGER-Lab/MMLU-Pro"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"
N_SHOTS = 5
HOLDOUT_PER_SUBJECT = 10
MAX_MODEL_LENGTH = 4096
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.8
SPLIT_SAMPLE_LIMIT: int | None = None
VAL_MAX_EXAMPLES_PER_SUBJECT = 10


@dataclass
class MMLUProEvaluationConfig(WhiteBoxEvaluationConfig):
    """MMLU-Pro evaluation config with prompt format options.

    Attributes:
        n_shots: Number of few-shot CoT examples per question. Set to 0 for zero-shot.
        use_chat_template: If True, format prompts as multi-turn chat messages using
            tokenizer.apply_chat_template(). If False, use raw text continuation format.
        chat_template_model: HuggingFace model ID to load a chat template from when the
            primary tokenizer has no chat_template. Useful for evaluating base models
            with their corresponding instruct model's chat template (e.g. using
            ``meta-llama/Llama-3.1-8B-Instruct``'s template for
            ``meta-llama/Meta-Llama-3-8B``). Only used when ``use_chat_template=True``.
    """

    n_shots: int = N_SHOTS
    use_chat_template: bool = False
    chat_template_model: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.n_shots < 0:
            raise ValueError(f"n_shots={self.n_shots} must be non-negative.")
        if self.n_shots > HOLDOUT_PER_SUBJECT:
            raise ValueError(
                f"n_shots={self.n_shots} exceeds HOLDOUT_PER_SUBJECT={HOLDOUT_PER_SUBJECT}. "
                f"Increase HOLDOUT_PER_SUBJECT or reduce n_shots."
            )


class _MMLUProVLLMBase(WhiteBoxEvaluation[MMLUProEvaluationConfig]):
    name: EvalName
    objective: MetricName = MetricName.MMLU_PRO_ACCURACY
    attacker_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        val_rows = self._load_split(VAL_SPLIT)
        test_rows = self._load_split(TEST_SPLIT)

        shot_pool = self._build_shot_pool(val_rows, test_rows)
        eval_rows = self._build_eval_items(val_rows, test_rows)

        self._cached_eval_rows: list[dict] = list(eval_rows)

        payload: pl.DataFrame = run_in_isolation(
            target=compute_mmlu_inferences,
            args=(
                self.eval_config,
                val_rows,
                test_rows,
                shot_pool,
                eval_rows,
            ),
            error_context="MMLU-Pro inference",
        )

        return InferenceSchema.validate(payload)

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        eval_rows = getattr(self, "_cached_eval_rows", None)
        if not eval_rows or len(eval_rows) != len(inferences):
            val_rows = self._load_split(VAL_SPLIT)
            test_rows = self._load_split(TEST_SPLIT)
            eval_rows = self._build_eval_items(val_rows, test_rows)

        preds: list[str] = []
        scores: list[float] = []

        for resp, row in zip(inferences[InferenceSchema.response], eval_rows, strict=False):
            response_text = str(resp)
            match = re.search(r"PRED:\s*([A-P])", response_text)
            if match:
                pred_letter = match.group(1)
            else:
                pred_letter = mmlu_api.extract_answer(response_text) or ""
            preds.append(pred_letter or "")
            gold = self._gold_letter(row)
            scores.append(1.0 if pred_letter == gold else 0.0)

        df = pl.from_dict(
            {
                ScoreSchema.prompt: list(inferences[InferenceSchema.prompt]),
                ScoreSchema.response: [
                    f"{resp}\nPRED: {pred}"
                    for resp, pred in zip(inferences[InferenceSchema.response], preds, strict=False)
                ],
                ScoreSchema.score: scores,
            }
        )
        return ScoreSchema.validate(df)

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        acc = float(pl.Series(scores[ScoreSchema.score]).mean())
        df = pl.from_dict(
            {
                EvaluationSchema.metric_name: [str(MetricName.MMLU_PRO_ACCURACY)],
                EvaluationSchema.metric_value: [float(acc) if acc == acc else 0.0],
            }
        )
        return EvaluationSchema.validate(df)

    def _load_split(self, split_name: str) -> list[dict]:
        split = (
            split_name
            if SPLIT_SAMPLE_LIMIT is None or SPLIT_SAMPLE_LIMIT <= 0
            else f"{split_name}[:{SPLIT_SAMPLE_LIMIT}]"
        )
        ds = datasets.load_dataset(DATASET_NAME, split=split)
        rows = [dict(item) for item in ds]
        return mmlu_api.preprocess(rows)

    def _build_eval_items(self, val_rows: list[dict], test_rows: list[dict]) -> list[dict]:
        raise NotImplementedError

    def _build_shot_pool(self, val_rows: list[dict], test_rows: list[dict]) -> dict[str, list[dict]]:
        return self._build_shot_pool_from_rows(val_rows)

    def _build_shot_pool_from_rows(self, rows: list[dict]) -> dict[str, list[dict]]:
        by_subj: dict[str, list[dict]] = {}
        for row in rows:
            subj = subject_key(row)
            by_subj.setdefault(subj, []).append(row)
        pool: dict[str, list[dict]] = {}
        for subj, rows in by_subj.items():
            holdout_count = min(
                HOLDOUT_PER_SUBJECT,
                self.eval_config.n_shots,
                len(rows),
            )
            pool[subj] = rows[:holdout_count]
        return pool

    def _gold_letter(self, row: dict) -> str:
        # Prefer explicit letter if present
        ans = row.get("answer")
        if isinstance(ans, str) and re.search(r"^[A-P]$", ans.strip(), flags=re.IGNORECASE):
            return ans.strip().upper()
        # Otherwise infer from index
        idx = row.get("answer_index")
        if isinstance(idx, int):
            letters = [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
            ]
            if 0 <= idx < len(letters):
                return letters[idx]
        # Fallback to A
        return "A"


@register_evaluation(EvalName.MMLU_PRO_VAL)
class MMLUProValEvaluation(_MMLUProVLLMBase):
    """Evaluate on a subset of the test split using few-shot exemplars drawn from the validation split."""

    name: EvalName = EvalName.MMLU_PRO_VAL

    def _build_eval_items(self, val_rows: list[dict], test_rows: list[dict]) -> list[dict]:
        return limit_examples_per_subject(test_rows, VAL_MAX_EXAMPLES_PER_SUBJECT)

    def _build_shot_pool(self, val_rows: list[dict], test_rows: list[dict]) -> dict[str, list[dict]]:
        return self._build_shot_pool_from_rows(val_rows)


@register_evaluation(EvalName.MMLU_PRO_TEST)
class MMLUProTestEvaluation(_MMLUProVLLMBase):
    """Evaluate on test split; shots come from validation split per subject."""

    name: EvalName = EvalName.MMLU_PRO_TEST

    def _build_eval_items(self, val_rows: list[dict], test_rows: list[dict]) -> list[dict]:
        return list(test_rows)

    def _build_shot_pool(self, val_rows: list[dict], test_rows: list[dict]) -> dict[str, list[dict]]:
        return self._build_shot_pool_from_rows(val_rows)


def compute_mmlu_inferences(
    eval_config: MMLUProEvaluationConfig,
    val_rows,
    test_rows,
    shot_pool,
    eval_rows,
) -> pl.DataFrame:
    """Run model on MMLU-Pro dataset to compute inferences.

    Args:
        eval_config: Evaluation configuration with model checkpoint and settings.
        val_rows: Validation split rows.
        test_rows: Test split rows.
        shot_pool: Few-shot exemplars by subject.
        eval_rows: Rows to evaluate on.

    Returns:
        A polars DataFrame with prompts and model responses.
    """
    torch.cuda.empty_cache()

    hf_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=(eval_config.model_config.tokenizer_checkpoint or eval_config.model_checkpoint),
        trust_remote_code=True,
    )

    if eval_config.use_chat_template:
        if not hf_tokenizer.chat_template:
            if eval_config.chat_template_model:
                fallback_tokenizer = AutoTokenizer.from_pretrained(
                    eval_config.chat_template_model, trust_remote_code=True
                )
                hf_tokenizer.chat_template = fallback_tokenizer.chat_template
            else:
                raise ValueError(
                    f"Tokenizer for {eval_config.model_checkpoint} has no chat_template and "
                    f"no chat_template_model was provided."
                )
        elif eval_config.chat_template_model:
            raise ValueError(
                f"chat_template_model={eval_config.chat_template_model!r} was specified but the tokenizer for "
                f"{eval_config.model_checkpoint} already has a chat_template."
            )
    # vllm tensor_parallel_size must be a factor of 64 (model hidden dimensions).
    # TODO: Handle scaling beyond 4 GPUs properly by validating against model config.
    llm_kwargs = {
        "model": eval_config.model_checkpoint,
        "tensor_parallel_size": torch.cuda.device_count() if torch.cuda.device_count() != 5 else 4,
        "gpu_memory_utilization": 0.8,
        "trust_remote_code": True,
        "max_model_len": MAX_MODEL_LENGTH,
    }

    # Use separate tokenizer if specified
    if eval_config.model_config.tokenizer_checkpoint is not None:
        llm_kwargs["tokenizer"] = eval_config.model_config.tokenizer_checkpoint

    llm = LLM(**llm_kwargs)

    max_new_tokens = min(eval_config.model_config.max_generation_length, MAX_NEW_TOKENS)
    sampling = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=max_new_tokens,
    )

    prompts: list[str] = []
    subjects: list[str] = []

    for row in tqdm(eval_rows):
        subject = subject_key(row)
        shots = shot_pool.get(subject)
        if shots is None or not shots:
            shots = val_rows if val_rows else test_rows

        k = eval_config.n_shots
        prompt: str | None = None
        while True:
            if eval_config.use_chat_template:
                messages = mmlu_api.generate_chat_messages(shots, row, max(k, 0))
                prompt = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = mmlu_api.generate_continuation_prompt(shots, row, max(k, 0))
            tokenized = hf_tokenizer(prompt, return_tensors="pt")
            length = int(tokenized["input_ids"].shape[-1])
            if length < (MAX_MODEL_LENGTH - max_new_tokens) or k <= 0:
                break
            k -= 1
        assert prompt is not None

        prompts.append(prompt)
        subjects.append(subject)

    preds, responses = mmlu_api.batch_inference(llm, sampling, prompts)

    inferences_dataframe = pl.from_dict(
        {
            InferenceSchema.prompt: prompts,
            InferenceSchema.response: [
                f"{resp}\nPRED: {pred or ''}" for resp, pred in zip(responses, preds, strict=False)
            ],
            "subject": subjects,
        }
    )

    return InferenceSchema.validate(inferences_dataframe)


def subject_key(row: dict) -> str:
    """Extract the subject/category key from a row."""
    return str(row.get("category") or row.get("subject") or "").strip()


def limit_examples_per_subject(rows: list[dict], max_examples: int) -> list[dict]:
    """Limit the number of examples per subject to max_examples."""
    if max_examples <= 0:
        return []
    selected: list[dict] = []
    seen_counts: dict[str, int] = {}
    for row in rows:
        subj = subject_key(row)
        count = seen_counts.get(subj, 0)
        if count < max_examples:
            selected.append(row)
            seen_counts[subj] = count + 1
    return selected
