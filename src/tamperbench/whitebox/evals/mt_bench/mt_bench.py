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
from torch import Tensor
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils_base import BatchEncoding
from typing_extensions import override

from tamperbench.whitebox.evals.base import (
    WhiteBoxEvaluation,
    WhiteBoxEvaluationConfig,
    load_hf_model_and_tokenizer,
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
from tamperbench.whitebox.utils import (
    EvalName,
    MetricName,
    OptimizationDirection,
    dealloc_model_and_tokenizer,
)

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

    def _format_prompt(self, messages: list[dict[str, str]], tokenizer: PreTrainedTokenizer) -> str:
        """Format messages into a prompt string using the tokenizer's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tokenizer: The tokenizer to use for formatting.

        Returns:
            str: Formatted prompt string.
        """
        if hasattr(tokenizer, "apply_chat_template"):
            result = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    def _generate_response(
        self, prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_new_tokens: int = 512
    ) -> str:
        """Generate a response from the model.

        Args:
            prompt: The input prompt.
            model: The model to use.
            tokenizer: The tokenizer to use.
            max_new_tokens: Maximum new tokens to generate.

        Returns:
            str: The generated response.
        """
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

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run model on MT-Bench questions to compute multi-turn inferences.

        Returns:
            DataFrame[InferenceSchema]: A polars DataFrame with prompts and responses.
        """
        model, tokenizer = load_hf_model_and_tokenizer(self.eval_config)

        inferences: list[dict[str, Any]] = []
        for question in tqdm(self.questions, desc="Generating MT-Bench responses"):
            # Build messages for turn 1
            messages_t1: list[dict[str, str]] = []
            if self.eval_config.system_prompt:
                messages_t1.append({"role": "system", "content": self.eval_config.system_prompt})
            messages_t1.append({"role": "user", "content": question.turns[0]})

            # Generate turn 1 response
            prompt_t1 = self._format_prompt(messages_t1, tokenizer)
            response_t1 = self._generate_response(
                prompt_t1, model, tokenizer, self.eval_config.model_config.max_generation_length
            )

            # Build messages for turn 2
            messages_t2 = [
                *messages_t1,
                {"role": "assistant", "content": response_t1},
                {"role": "user", "content": question.turns[1]},
            ]

            # Generate turn 2 response
            prompt_t2 = self._format_prompt(messages_t2, tokenizer)
            response_t2 = self._generate_response(
                prompt_t2, model, tokenizer, self.eval_config.model_config.max_generation_length
            )

            # Store inference record
            inferences.append(
                {
                    "question_id": question.question_id,
                    "category": question.category,
                    "turn_1_question": question.turns[0],
                    "turn_1_response": response_t1,
                    "turn_2_question": question.turns[1],
                    "turn_2_response": response_t2,
                    InferenceSchema.prompt: prompt_t2,
                    InferenceSchema.response: response_t2,
                }
            )

        dealloc_model_and_tokenizer(model, tokenizer)
        return InferenceSchema.validate(pl.DataFrame(inferences))

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
