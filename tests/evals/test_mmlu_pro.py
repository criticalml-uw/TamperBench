"""Sanity check for MMLU-Pro evaluator."""

import tempfile
from typing import Any

import pytest
from dotenv import load_dotenv

from tamperbench.whitebox.evals import MMLUProEvaluationConfig, MMLUProValEvaluation
from tamperbench.whitebox.evals.mmlu_pro.eval_from_api import (
    generate_chat_messages,
    generate_continuation_prompt,
)
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import MetricName, TemplateName

SUBJECT = "biology"

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
    "The following are multiple choice questions (with answers) about biology."
    " Think step by step and then finish your answer with"
    ' "the answer is (X)" where X is the correct letter choice.'
)
CHAT_SYSTEM_INSTRUCTION = (
    "The following are multiple choice questions about biology."
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


@pytest.mark.expensive
def test_mmlu_pro_evaluation() -> None:
    """Test that MMLU-Pro evaluator produces valid results."""
    load_dotenv()  # ensure HF_TOKEN available

    with tempfile.TemporaryDirectory() as tmpdirname:
        llama_3_8b_eval_config = MMLUProEvaluationConfig(
            model_checkpoint="meta-llama/Llama-3.1-8B-Instruct",
            out_dir=tmpdirname,
            model_config=ModelConfig(
                template_name=TemplateName.NATIVE,
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
