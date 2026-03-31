"""Sanity check for MMLU-Pro evaluator."""

import contextlib
import tempfile
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest
from dotenv import load_dotenv

from tamperbench.whitebox.evals import MMLUProEvaluationConfig, MMLUProValEvaluation
from tamperbench.whitebox.evals.mmlu_pro.eval_from_api import (
    generate_chat_messages,
    generate_continuation_prompt,
)
from tamperbench.whitebox.evals.mmlu_pro.llm_judge import format_grading_prompt
from tamperbench.whitebox.evals.mmlu_pro.mmlu_pro import format_options
from tamperbench.whitebox.evals.output_schema import EvaluationSchema, InferenceSchema
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import MetricName

SUBJECT = "math"

MMLUProRow = dict[str, Any]
MMLUProSamples = tuple[list[MMLUProRow], MMLUProRow]


@pytest.fixture
def mmlu_pro_samples() -> MMLUProSamples:
    """Provide sample MMLU-Pro data: a shot pool (2 examples) and a target question."""
    shot_pool = [
        {
            "question": "shot0",
            "options": ["a", "b", "c"],
            "category": SUBJECT,
            "cot_content": ("A: Let's think step by step. 0 + 0 = 0. The answer is (A)."),
            "answer": "A",
            "answer_index": 0,
        },
        {
            "question": "shot1",
            "options": ["1", "2", "3"],
            "category": SUBJECT,
            "cot_content": ("A: Let's think step by step. 1 + 1 = 2. The answer is (B)."),
            "answer": "B",
            "answer_index": 1,
        },
    ]
    target = {
        "question": "q",
        "options": ["x", "y", "z"],
        "category": SUBJECT,
        "cot_content": "",
        "answer": "C",
        "answer_index": 2,
    }
    return shot_pool, target


# These constants contain what we expect mmlu_pro_samples to get formatted into.
CONTINUATION_SYSTEM_INSTRUCTION = (
    "The following are multiple choice questions (with answers) about math."
    " Think step by step and then finish your answer with"
    ' "the answer is (X)" where X is the correct letter choice.'
)
CHAT_SYSTEM_INSTRUCTION = (
    "The following are multiple choice questions about math."
    " Think step by step and then finish your answer with"
    ' "the answer is (X)" where X is the correct letter choice.'
)
SHOT1_QUESTION_AND_OPTIONS = "Question:\nshot0\nOptions:\nA. a\nB. b\nC. c\n"
SHOT1_COT_CHAT = "Let's think step by step. 0 + 0 = 0. The answer is (A)."
SHOT1_COT_CONTINUATION = f"Answer: {SHOT1_COT_CHAT}\n\n"
SHOT2_QUESTION_AND_OPTIONS = "Question:\nshot1\nOptions:\nA. 1\nB. 2\nC. 3\n"
SHOT2_COT_CHAT = "Let's think step by step. 1 + 1 = 2. The answer is (B)."
SHOT2_COT_CONTINUATION = f"Answer: {SHOT2_COT_CHAT}\n\n"
TARGET_QUESTION_AND_OPTIONS = "Question:\nq\nOptions:\nA. x\nB. y\nC. z\n"
TARGET_COT_SUFFIX = "Answer: Let's think step by step."


def test_fewshot_continuation_prompt(mmlu_pro_samples: MMLUProSamples) -> None:
    """Few-shot continuation prompt matches exact expected text."""
    shot_pool, target = mmlu_pro_samples
    prompt = generate_continuation_prompt(shot_pool, target, k=2)

    expected = (
        CONTINUATION_SYSTEM_INSTRUCTION
        + "\n\n"
        + SHOT1_QUESTION_AND_OPTIONS
        + SHOT1_COT_CONTINUATION
        + SHOT2_QUESTION_AND_OPTIONS
        + SHOT2_COT_CONTINUATION
        + TARGET_QUESTION_AND_OPTIONS
        + TARGET_COT_SUFFIX
    )
    assert prompt == expected


def test_zeroshot_continuation_prompt(mmlu_pro_samples: MMLUProSamples) -> None:
    """Zero-shot continuation prompt matches exact expected text."""
    shot_pool, target = mmlu_pro_samples
    prompt = generate_continuation_prompt(shot_pool, target, k=0)

    expected = CONTINUATION_SYSTEM_INSTRUCTION + "\n\n" + TARGET_QUESTION_AND_OPTIONS + TARGET_COT_SUFFIX
    assert prompt == expected


def test_fewshot_chat_messages(mmlu_pro_samples: MMLUProSamples) -> None:
    """Few-shot chat messages match exact expected structure and content."""
    shot_pool, target = mmlu_pro_samples
    messages = generate_chat_messages(shot_pool, target, k=2)

    expected = [
        {"role": "system", "content": CHAT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": SHOT1_QUESTION_AND_OPTIONS.strip()},
        {"role": "assistant", "content": SHOT1_COT_CHAT},
        {"role": "user", "content": SHOT2_QUESTION_AND_OPTIONS.strip()},
        {"role": "assistant", "content": SHOT2_COT_CHAT},
        {"role": "user", "content": TARGET_QUESTION_AND_OPTIONS.strip()},
    ]
    assert messages == expected


def test_zeroshot_chat_messages(mmlu_pro_samples: MMLUProSamples) -> None:
    """Zero-shot chat messages match exact expected structure and content."""
    shot_pool, target = mmlu_pro_samples
    messages = generate_chat_messages(shot_pool, target, k=0)

    expected = [
        {"role": "system", "content": CHAT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": TARGET_QUESTION_AND_OPTIONS.strip()},
    ]
    assert messages == expected


@contextlib.contextmanager
def _make_eval(eval_rows: list[MMLUProRow], scoring_method: str = "regex") -> Generator[MMLUProValEvaluation]:
    """Create a MMLUProValEvaluation with cached eval_rows (no HF download).

    Injects ``_cached_eval_rows`` directly so that ``compute_scores`` skips
    the HuggingFace dataset download that normally populates eval rows during
    ``compute_inferences``.  This couples the test to an implementation detail,
    but the alternative is downloading the dataset in every unit test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MMLUProEvaluationConfig(
            model_checkpoint="dummy",
            out_dir=tmpdir,
            model_config=ModelConfig(
                user_prefix="",
                assistant_prefix="",
                end_turn="",
                max_generation_length=128,
                inference_batch_size=1,
            ),
            scoring_method=scoring_method,  # pyright: ignore[reportArgumentType]
        )
        evaluator = MMLUProValEvaluation(config)
        evaluator._cached_eval_rows = list(eval_rows)  # pyright: ignore[reportPrivateUsage]
        yield evaluator


def _make_inferences(responses: list[str]) -> pl.DataFrame:
    return InferenceSchema.validate(
        pl.from_dict(
            {
                InferenceSchema.prompt: [f"prompt_{i}" for i in range(len(responses))],
                InferenceSchema.response: responses,
            }
        )
    )


def test_regex_scoring_method(mmlu_pro_samples: MMLUProSamples) -> None:
    """Regex scoring extracts letter answers and compares to gold."""
    shot_pool, target = mmlu_pro_samples
    eval_rows = [*shot_pool, target]
    with _make_eval(eval_rows, "regex") as evaluator:
        inferences = _make_inferences(
            [
                "The answer is (A)",  # correct
                "I think the answer is B",  # correct
                "The answer is (A)",  # incorrect (correct is C)
            ]
        )
        scores_df = evaluator.compute_scores(inferences)  # pyright: ignore[reportArgumentType]
        scores = scores_df["score"].to_list()
        assert scores == [1.0, 1.0, 0.0]


def test_judge_scoring_method(mmlu_pro_samples: MMLUProSamples) -> None:
    """Judge scoring calls llm_judge and returns scores based on judge verdicts."""
    shot_pool, target = mmlu_pro_samples
    eval_rows = [*shot_pool, target]
    with _make_eval(eval_rows, "judge") as evaluator:
        call_count = 0

        def fake_grade(**kwargs: Any) -> float:
            nonlocal call_count
            call_count += 1
            # Mark correct when the response contains the right answer text
            model_response = kwargs.get("model_response", "")
            if "answer_0" in model_response or "answer_z" in model_response:
                return 1.0
            return 0.0

        inferences = _make_inferences(
            [
                "answer_0",  # correct
                "wrong",  # wrong
                "answer_z",  # correct
            ]
        )

        with (
            patch("tamperbench.whitebox.evals.mmlu_pro.mmlu_pro.OpenAI"),
            patch("tamperbench.whitebox.evals.mmlu_pro.mmlu_pro.llm_judge.grade_with_context", side_effect=fake_grade),
        ):
            scores_df = evaluator.compute_scores(inferences)  # pyright: ignore[reportArgumentType]

        scores = scores_df["score"].to_list()
        assert scores == [1.0, 0.0, 1.0]
        assert call_count == 3


def test_judge_prompt(mmlu_pro_samples: MMLUProSamples) -> None:
    """Judge grading prompt matches expected format."""
    _shot_pool, target = mmlu_pro_samples

    options_text = format_options(target)
    assert options_text == "A. x\nB. y\nC. z"

    prompt = format_grading_prompt(
        question="q",
        choices=options_text,
        ground_truth_answer="C",
        model_response="I think the answer is z.",
    )
    assert (
        prompt
        == """\
## Question
q

## Answer Choices
A. x
B. y
C. z

## Correct Answer
C

## Model's Response
I think the answer is z."""
    )


@pytest.mark.expensive
def test_mmlu_pro_evaluation() -> None:
    """Test that MMLU-Pro evaluator produces valid results."""
    load_dotenv()  # ensure HF_TOKEN available

    with tempfile.TemporaryDirectory() as tmpdirname:
        llama_3_8b_eval_config = MMLUProEvaluationConfig(
            model_checkpoint="meta-llama/Llama-3.1-8B-Instruct",
            out_dir=tmpdirname,
            model_config=ModelConfig(
                user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
                assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
                end_turn="<|eot_id|>\n",
                max_generation_length=1024,
                inference_batch_size=16,
            ),
        )

        llama_3_8b_evaluation = MMLUProValEvaluation(llama_3_8b_eval_config)

        llama_3_8b_eval: dict[str, tuple[float]] = llama_3_8b_evaluation.run_evaluation().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )

        accuracy = llama_3_8b_eval[MetricName.MMLU_PRO_ACCURACY][0]
        assert 0.0 <= accuracy <= 1.0

        print("Llama3-8B Instruct MMLU-Pro Val:", llama_3_8b_eval)
