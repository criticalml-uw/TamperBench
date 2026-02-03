"""MT-Bench evaluation using LLM-as-a-Judge.

Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf
"""

from dataclasses import dataclass
from functools import partial

import polars as pl
import torch
from openai import OpenAI
from pandera.typing.polars import DataFrame
from torch import Tensor
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils_base import BatchEncoding
from typing_extensions import override

from tamperbench.whitebox.evals.base import (
    WhiteBoxEvaluation,
    WhiteBoxEvaluationConfig,
    load_model_and_tokenizer,
)
from tamperbench.whitebox.evals.mt_bench.judge import (
    score_multi_turn,
    score_single_turn,
)
from tamperbench.whitebox.evals.mt_bench.questions import (
    load_mt_bench_questions_by_category,
)
from tamperbench.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from tamperbench.whitebox.utils import (
    EvalName,
    MetricName,
    OptimizationDirection,
    dealloc_model_and_tokenizer,
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

    judge_model: str = "gpt-4o"
    categories: list[str] | None = None
    system_prompt: str | None = None
    max_judge_workers: int = 16


# Schema for MT-Bench multi-turn inferences
class MTBenchInferenceSchema(InferenceSchema):
    """Schema for MT-Bench multi-turn inferences."""

    question_id: int
    category: str
    turn_1_question: str
    turn_1_response: str
    turn_2_question: str
    turn_2_response: str


class MTBenchScoreSchema(ScoreSchema):
    """Schema for MT-Bench scores with per-turn details."""

    question_id: int
    category: str
    turn_1_score: float
    turn_2_score: float
    judge_response_1: str
    judge_response_2: str


class MTBenchEvaluation(WhiteBoxEvaluation):
    """MT-Bench evaluation using LLM-as-a-Judge."""

    name: EvalName = EvalName.MT_BENCH
    objective: MetricName = MetricName.MT_BENCH_SCORE
    attacker_direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    def __init__(self, eval_config: MTBenchEvaluationConfig) -> None:
        """Initialize MT-Bench evaluation.

        Args:
            eval_config: Configuration for the evaluation.
        """
        super().__init__(eval_config)
        self.openai_client = OpenAI()
        self.questions = load_mt_bench_questions_by_category(eval_config.categories)

    def _format_prompt(self, messages: list[dict[str, str]], tokenizer) -> str:
        """Format messages into a prompt string using the tokenizer's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tokenizer: The tokenizer to use for formatting.

        Returns:
            str: Formatted prompt string.
        """
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    def _generate_response(self, prompt: str, model, tokenizer, max_new_tokens: int = 512) -> str:
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

        input_ids: Tensor = encoding["input_ids"].to(model.device)
        attention_mask: Tensor = encoding["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs: GenerateOutput = model.generate(
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
        # Standard inference path
        model, tokenizer = load_model_and_tokenizer(self.eval_config)

        inferences: list[dict] = []
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

    def _score_row(self, row: dict, question_map: dict) -> dict:
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
            ScoreSchema.score: (score_1 + score_2) / 2,
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
        mean_score: float = float(scores_df[ScoreSchema.score].mean())

        # Per-turn averages
        mean_turn_1: float = float(scores_df["turn_1_score"].mean())
        mean_turn_2: float = float(scores_df["turn_2_score"].mean())

        _metrics_df: pl.DataFrame = pl.from_dict(
            data={
                EvaluationSchema.metric_name: [
                    str(MetricName.MT_BENCH_SCORE),
                    str(MetricName.MT_BENCH_TURN_1_SCORE),
                    str(MetricName.MT_BENCH_TURN_2_SCORE),
                ],
                EvaluationSchema.metric_value: [mean_score, mean_turn_1, mean_turn_2],
            }
        )
        return EvaluationSchema.validate(_metrics_df)
