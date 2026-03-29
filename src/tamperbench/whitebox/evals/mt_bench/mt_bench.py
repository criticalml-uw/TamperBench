"""MT-Bench evaluation using LLM-as-a-Judge.

Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import pandera.polars as pa
import polars as pl
import torch
from openai import OpenAI
from pandera.typing.polars import DataFrame
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from typing_extensions import override

from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils_base import BatchEncoding

from tamperbench.whitebox.evals.base import (
    WhiteBoxEvaluation,
    WhiteBoxEvaluationConfig,
    load_hf_model_and_tokenizer,
    load_vllm_model_and_tokenizer,
)
from tamperbench.whitebox.evals.mt_bench.judge import (
    score_multi_turn,
    score_single_turn,
)
from tamperbench.whitebox.evals.mt_bench.questions import (
    MTBenchQuestion,
    load_mt_bench_questions_by_category,
)
from tamperbench.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from tamperbench.whitebox.evals.reference import ReferenceScore, ReferenceScores
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.evals.utils import format_chat_prompt, generate
from tamperbench.whitebox.utils import (
    EvalName,
    MetricName,
    OptimizationDirection,
    dealloc_model_and_tokenizer,
)
from tamperbench.whitebox.utils.ops import run_in_isolation

# Reference: Multiple sources for MT-Bench scores
# Primary: "Judging LLM-as-a-Judge" (Zheng et al., NeurIPS 2023)
# https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf
# Additional: WMDP Benchmark (Li et al., 2024) Table 2
# MT-Bench scores are on a 1-10 scale (higher = better)
MT_BENCH_REFERENCE_SCORES = ReferenceScores(
    scores=[
        # From WMDP paper Table 2
        ReferenceScore(
            model_id="HuggingFaceH4/zephyr-7b-beta",
            metrics={
                str(MetricName.MT_BENCH_SCORE): 7.33,
            },
            source="WMDP Benchmark (Li et al., 2024) Table 2",
            tolerance=0.10,  # ~1 point tolerance for subjective scoring
            notes="Single evaluation mode, GPT-4-0613 judge",
        ),
        ReferenceScore(
            model_id="01-ai/Yi-34B-Chat",
            metrics={
                str(MetricName.MT_BENCH_SCORE): 7.65,
            },
            source="WMDP Benchmark (Li et al., 2024) Table 2",
            tolerance=0.10,
        ),
        ReferenceScore(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            metrics={
                str(MetricName.MT_BENCH_SCORE): 8.30,
            },
            source="WMDP Benchmark (Li et al., 2024) Table 2",
            tolerance=0.10,
        ),
        ReferenceScore(
            model_id="gpt-4",
            metrics={
                str(MetricName.MT_BENCH_SCORE): 9.13,
            },
            source="WMDP Benchmark (Li et al., 2024) Table 2",
            tolerance=0.10,
            notes="OpenAI GPT-4 (2023a)",
        ),
        # From original MT-Bench paper (Zheng et al., 2023) - Table 5
        ReferenceScore(
            model_id="gpt-4-0314",
            metrics={
                str(MetricName.MT_BENCH_SCORE): 8.99,
                str(MetricName.MT_BENCH_TURN_1_SCORE): 8.96,
                str(MetricName.MT_BENCH_TURN_2_SCORE): 9.03,
            },
            source="MT-Bench (Zheng et al., 2023) Table 5",
            tolerance=0.10,
            notes="Original paper GPT-4 results",
        ),
        ReferenceScore(
            model_id="gpt-3.5-turbo",
            metrics={
                str(MetricName.MT_BENCH_SCORE): 7.94,
                str(MetricName.MT_BENCH_TURN_1_SCORE): 8.08,
                str(MetricName.MT_BENCH_TURN_2_SCORE): 7.81,
            },
            source="MT-Bench (Zheng et al., 2023) Table 5",
            tolerance=0.10,
        ),
        ReferenceScore(
            model_id="claude-v1",
            metrics={
                str(MetricName.MT_BENCH_SCORE): 7.90,
                str(MetricName.MT_BENCH_TURN_1_SCORE): 8.15,
                str(MetricName.MT_BENCH_TURN_2_SCORE): 7.65,
            },
            source="MT-Bench (Zheng et al., 2023) Table 5",
            tolerance=0.10,
        ),
        ReferenceScore(
            model_id="meta-llama/Llama-2-70b-chat-hf",
            metrics={
                str(MetricName.MT_BENCH_SCORE): 6.86,
                str(MetricName.MT_BENCH_TURN_1_SCORE): 7.18,
                str(MetricName.MT_BENCH_TURN_2_SCORE): 6.54,
            },
            source="MT-Bench (Zheng et al., 2023) Table 5",
            tolerance=0.10,
        ),
        ReferenceScore(
            model_id="lmsys/vicuna-33b-v1.3",
            metrics={
                str(MetricName.MT_BENCH_SCORE): 7.12,
                str(MetricName.MT_BENCH_TURN_1_SCORE): 7.46,
                str(MetricName.MT_BENCH_TURN_2_SCORE): 6.79,
            },
            source="MT-Bench (Zheng et al., 2023) Table 5",
            tolerance=0.10,
        ),
        ReferenceScore(
            model_id="WizardLM/WizardLM-13B-V1.2",
            metrics={
                str(MetricName.MT_BENCH_SCORE): 7.20,
                str(MetricName.MT_BENCH_TURN_1_SCORE): 7.50,
                str(MetricName.MT_BENCH_TURN_2_SCORE): 6.90,
            },
            source="MT-Bench (Zheng et al., 2023) Table 5",
            tolerance=0.10,
        ),
    ]
)


@dataclass
class MTBenchEvaluationConfig(WhiteBoxEvaluationConfig):
    """Configuration for MT-Bench evaluation.

    Attributes:
        judge_model: Model to use for LLM-as-a-Judge scoring (default: gpt-4o).
        categories: List of MT-Bench categories to evaluate. If None, use all.
        system_prompt: Optional system prompt for the model being evaluated.
        max_judge_workers: Max parallel threads for LLM judge API calls.
    """

    # NOTE: The original MT-Bench paper and WMDP reference scores used gpt-4-0613 as
    # the judge, which is now deprecated. Default is gpt-4o; scores may differ slightly
    # from reference values. Set judge_model="gpt-4-0613" if you have legacy API access
    # and need exact reproduction of paper results.
    judge_model: str = "gpt-4o"
    categories: list[str] | None = None
    system_prompt: str | None = None
    max_judge_workers: int = 16


# Schema for MT-Bench multi-turn inferences
class MTBenchInferenceSchema(InferenceSchema):
    """Schema for MT-Bench multi-turn inferences."""

    question_id: int = cast("int", pa.Field(nullable=False))
    category: str = cast("str", pa.Field(nullable=False))
    turn_1_question: str = cast("str", pa.Field(nullable=False))
    turn_1_response: str = cast("str", pa.Field(nullable=False))
    turn_2_question: str = cast("str", pa.Field(nullable=False))
    turn_2_response: str = cast("str", pa.Field(nullable=False))


class MTBenchScoreSchema(ScoreSchema):
    """Schema for MT-Bench scores with per-turn details."""

    question_id: int = cast("int", pa.Field(nullable=False))
    category: str = cast("str", pa.Field(nullable=False))
    turn_1_score: float = cast("float", pa.Field(nullable=False, coerce=True))
    turn_2_score: float = cast("float", pa.Field(nullable=False, coerce=True))
    judge_response_1: str = cast("str", pa.Field(nullable=False))
    judge_response_2: str = cast("str", pa.Field(nullable=False))


@register_evaluation(EvalName.MT_BENCH, MTBenchEvaluationConfig)
class MTBenchEvaluation(WhiteBoxEvaluation[MTBenchEvaluationConfig]):
    """MT-Bench evaluation using LLM-as-a-Judge."""

    name: EvalName = EvalName.MT_BENCH
    objective: MetricName = MetricName.MT_BENCH_SCORE
    attacker_direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    reference_scores: ReferenceScores = MT_BENCH_REFERENCE_SCORES

    def __init__(self, eval_config: MTBenchEvaluationConfig) -> None:
        """Initialize MT-Bench evaluation.

        Args:
            eval_config: Configuration for the evaluation.
        """
        super().__init__(eval_config)
        self.openai_client: OpenAI = OpenAI()
        self.questions: list[MTBenchQuestion] = load_mt_bench_questions_by_category(eval_config.categories)

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run model on MT-Bench questions to compute multi-turn inferences.

        Uses vLLM with batched generation: all turn-1 prompts are generated in
        one batch, then all turn-2 prompts (which depend on turn-1 responses)
        are generated in a second batch.

        Returns:
            DataFrame[InferenceSchema]: A polars DataFrame with prompts and responses.
        """
        # Serialize questions for the isolated process
        questions_data = [
            {
                "question_id": q.question_id,
                "category": q.category,
                "turns": q.turns,
            }
            for q in self.questions
        ]

        payload: pl.DataFrame = run_in_isolation(
            target=_instantiate_model_and_infer,
            args=(self.eval_config, questions_data),
            error_context="MT-Bench inference",
        )

        return InferenceSchema.validate(payload)

    def _score_row(self, row: dict[str, Any], question_map: dict[int, MTBenchQuestion]) -> dict[str, Any]:
        """Score a single inference row (both turns).

        Args:
            row: Row from inferences DataFrame.
            question_map: Map from question_id to MTBenchQuestion.

        Returns:
            dict: Scored row.
        """
        question = question_map.get(row["question_id"])
        reference = question.reference if question else None

        ref_1 = reference[0] if reference else None
        score_1, judge_resp_1 = score_single_turn(
            client=self.openai_client,
            judge_model=self.eval_config.judge_model,
            question=row["turn_1_question"],
            answer=row["turn_1_response"],
            reference=ref_1,
            category=row["category"],
        )

        ref_2 = reference[1] if reference and len(reference) > 1 else None
        score_2, judge_resp_2 = score_multi_turn(
            client=self.openai_client,
            judge_model=self.eval_config.judge_model,
            question_1=row["turn_1_question"],
            answer_1=row["turn_1_response"],
            question_2=row["turn_2_question"],
            answer_2=row["turn_2_response"],
            reference_1=ref_1,
            reference_2=ref_2,
            category=row["category"],
        )

        return {
            InferenceSchema.prompt: row[InferenceSchema.prompt],
            InferenceSchema.response: row[InferenceSchema.response],
            ScoreSchema.score: (score_1 + score_2) / 2,  # pyright: ignore[reportReturnType]
            "question_id": row["question_id"],
            "category": row["category"],
            "turn_1_score": score_1,
            "turn_2_score": score_2,
            "judge_response_1": judge_resp_1,
            "judge_response_2": judge_resp_2,
        }

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Use LLM-as-a-Judge to score model responses.

        Args:
            inferences: DataFrame with model inferences.

        Returns:
            DataFrame[ScoreSchema]: DataFrame with scored inferences.
        """
        question_map = {q.question_id: q for q in self.questions}
        rows = list(inferences.iter_rows(named=True))

        scores = thread_map(
            partial(self._score_row, question_map=question_map),
            rows,
            max_workers=self.eval_config.max_judge_workers,
            desc="Scoring with LLM judge",
        )

        return ScoreSchema.validate(pl.DataFrame(scores))

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        """Aggregate MT-Bench scores into evaluation metrics.

        Args:
            scores: DataFrame with per-question scores.

        Returns:
            DataFrame[EvaluationSchema]: DataFrame with aggregated metrics.
        """
        scores_df = ScoreSchema.validate(scores)

        # Overall average score
        mean_score: float = float(scores_df[ScoreSchema.score].mean())  # pyright: ignore[reportArgumentType,reportCallIssue]

        # Per-turn averages
        mean_turn_1: float = float(scores_df["turn_1_score"].mean())  # pyright: ignore[reportArgumentType]
        mean_turn_2: float = float(scores_df["turn_2_score"].mean())  # pyright: ignore[reportArgumentType]

        _metrics_df: pl.DataFrame = pl.from_dict(
            data={
                EvaluationSchema.metric_name: [
                    str(MetricName.MT_BENCH_SCORE),
                    str(MetricName.MT_BENCH_TURN_1_SCORE),
                    str(MetricName.MT_BENCH_TURN_2_SCORE),
                ],
                EvaluationSchema.metric_value: [mean_score, mean_turn_1, mean_turn_2],  # pyright: ignore[reportArgumentType]
            }
        )
        return EvaluationSchema.validate(_metrics_df)


def _instantiate_model_and_infer(
    eval_config: MTBenchEvaluationConfig,
    questions_data: list[dict[str, Any]],
) -> pl.DataFrame:
    """Run batched vLLM inference for MT-Bench multi-turn questions.

    Turn-1 prompts are independent and batched together. Turn-2 prompts depend
    on turn-1 responses, so they are batched in a second pass.

    Args:
        eval_config: Evaluation configuration with model checkpoint and settings.
        questions_data: Serialized question dicts with question_id, category, turns.

    Returns:
        A polars DataFrame with multi-turn inference results.
    """
    model, tokenizer = load_vllm_model_and_tokenizer(eval_config)
    try:
        system_prompt = eval_config.system_prompt
        max_new_tokens = eval_config.model_config.max_generation_length

        # Disable thinking for MT-Bench: thinking tokens would confuse the LLM judge
        # and waste the generation budget on reasoning rather than the actual answer.

        # --- Turn 1: batch all questions ---
        turn_1_prompts = [
            format_chat_prompt(q["turns"][0], tokenizer, system_prompt=system_prompt, enable_thinking=False)
            for q in questions_data
        ]
        turn_1_responses = generate(turn_1_prompts, model, max_new_tokens=max_new_tokens, enable_thinking=False)

        # --- Turn 2: batch with turn-1 history ---
        turn_2_prompts = [
            format_chat_prompt(
                q["turns"][1],
                tokenizer,
                system_prompt=system_prompt,
                history=[
                    {"role": "user", "content": q["turns"][0]},
                    {"role": "assistant", "content": resp},
                ],
                enable_thinking=False,
            )
            for q, resp in zip(questions_data, turn_1_responses, strict=True)
        ]
        turn_2_responses = generate(turn_2_prompts, model, max_new_tokens=max_new_tokens, enable_thinking=False)

        # Build output DataFrame
        # Concatenate both turns into prompt/response for the base schema
        combined_prompts = [f"{p1}\n\n{p2}" for p1, p2 in zip(turn_1_prompts, turn_2_prompts, strict=True)]
        combined_responses = [f"{r1}\n\n{r2}" for r1, r2 in zip(turn_1_responses, turn_2_responses, strict=True)]

        rows: dict[str, list[Any]] = {
            InferenceSchema.prompt: combined_prompts,
            InferenceSchema.response: combined_responses,
            "question_id": [q["question_id"] for q in questions_data],
            "category": [q["category"] for q in questions_data],
            "turn_1_question": [q["turns"][0] for q in questions_data],
            "turn_1_response": turn_1_responses,
            "turn_2_question": [q["turns"][1] for q in questions_data],
            "turn_2_response": turn_2_responses,
        }

        return pl.from_dict(data=rows)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@register_evaluation(EvalName.MT_BENCH_HF, MTBenchEvaluationConfig)
class MTBenchHFEvaluation(MTBenchEvaluation):
    """MT-Bench evaluation using sequential HuggingFace generate.

    This variant uses HF transformers model.generate() instead of vLLM,
    processing questions one at a time. Useful for comparing inference
    outputs between vLLM and HF backends.
    """

    name: EvalName = EvalName.MT_BENCH_HF

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run model on MT-Bench questions using sequential HF generate.

        Returns:
            DataFrame[InferenceSchema]: A polars DataFrame with prompts and responses.
        """
        questions_data = [
            {
                "question_id": q.question_id,
                "category": q.category,
                "turns": q.turns,
            }
            for q in self.questions
        ]

        payload: pl.DataFrame = run_in_isolation(
            target=_instantiate_model_and_infer_hf,
            args=(self.eval_config, questions_data),
            error_context="MT-Bench HF inference",
        )

        return InferenceSchema.validate(payload)


def _instantiate_model_and_infer_hf(
    eval_config: MTBenchEvaluationConfig,
    questions_data: list[dict[str, Any]],
) -> pl.DataFrame:
    """Run sequential HF inference for MT-Bench multi-turn questions.

    Processes each question one at a time using HuggingFace model.generate().
    Slower than vLLM but produces the original HF generation outputs.

    Args:
        eval_config: Evaluation configuration with model checkpoint and settings.
        questions_data: Serialized question dicts with question_id, category, turns.

    Returns:
        A polars DataFrame with multi-turn inference results.
    """
    model, tokenizer = load_hf_model_and_tokenizer(eval_config)
    try:
        system_prompt = eval_config.system_prompt
        max_new_tokens = eval_config.model_config.max_generation_length

        inferences: list[dict[str, Any]] = []
        for q in tqdm(questions_data, desc="Generating MT-Bench responses (HF)"):
            # Build turn-1 messages
            messages_t1: list[dict[str, str]] = []
            if system_prompt:
                messages_t1.append({"role": "system", "content": system_prompt})
            messages_t1.append({"role": "user", "content": q["turns"][0]})

            prompt_t1 = _apply_chat_template(messages_t1, tokenizer)
            response_t1 = _hf_generate(prompt_t1, model, tokenizer, max_new_tokens)

            # Build turn-2 messages with turn-1 history
            messages_t2 = [
                *messages_t1,
                {"role": "assistant", "content": response_t1},
                {"role": "user", "content": q["turns"][1]},
            ]

            prompt_t2 = _apply_chat_template(messages_t2, tokenizer)
            response_t2 = _hf_generate(prompt_t2, model, tokenizer, max_new_tokens)

            inferences.append(
                {
                    InferenceSchema.prompt: f"{prompt_t1}\n\n{prompt_t2}",
                    InferenceSchema.response: f"{response_t1}\n\n{response_t2}",
                    "question_id": q["question_id"],
                    "category": q["category"],
                    "turn_1_question": q["turns"][0],
                    "turn_1_response": response_t1,
                    "turn_2_question": q["turns"][1],
                    "turn_2_response": response_t2,
                }
            )

        return pl.from_dict(
            {k: [row[k] for row in inferences] for k in inferences[0]}
        )
    finally:
        dealloc_model_and_tokenizer(model, tokenizer)


def _apply_chat_template(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
) -> str:
    """Apply chat template to messages, with fallback for models without one."""
    if hasattr(tokenizer, "apply_chat_template"):
        result = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert isinstance(result, str)
        return result
    # Fallback for models without chat template
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"System: {msg['content']}\n\n"
        elif msg["role"] == "user":
            prompt += f"User: {msg['content']}\n\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n\n"
    prompt += "Assistant: "
    return prompt


def _hf_generate(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 512,
) -> str:
    """Generate a single response using HuggingFace model.generate()."""
    encoding: BatchEncoding = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    input_ids: Tensor = cast(Tensor, encoding["input_ids"]).to(model.device)
    attention_mask: Tensor = cast(Tensor, encoding["attention_mask"]).to(model.device)

    with torch.no_grad():
        outputs: GenerateOutput = model.generate(  # pyright: ignore[reportCallIssue]
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][input_ids.shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response
