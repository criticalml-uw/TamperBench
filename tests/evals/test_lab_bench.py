"""Tests for LAB-Bench evaluation."""

import polars as pl
import pytest

from tamperbench.whitebox.evals.lab_bench.lab_bench import (
    INSUFFICIENT_INFO_OPTION,
    LabBenchEvaluation,
    LabBenchEvaluationConfig,
    LabBenchQuestion,
    LabBenchSubtask,
    build_choices,
    extract_answer,
    format_question,
)
from tamperbench.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from tamperbench.whitebox.utils import MetricName
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import TemplateName

# --- Tests for build_choices ---


def test_build_choices_includes_all_options_and_insufficient_info() -> None:
    """All distractors, the ideal answer, and 'Insufficient information' appear in the output."""
    ideal = "correct answer"
    distractors = ["wrong1", "wrong2", "wrong3"]
    choices, correct_idx = build_choices(ideal, distractors, seed="test")
    assert set(choices) == {ideal, *distractors, INSUFFICIENT_INFO_OPTION}
    assert choices[correct_idx] == ideal
    assert len(choices) == 5  # 1 ideal + 3 distractors + 1 insufficient info


def test_build_choices_deterministic() -> None:
    """Same seed produces the same shuffle."""
    ideal = "correct"
    distractors = ["a", "b", "c"]
    result1 = build_choices(ideal, distractors, seed="fixed")
    result2 = build_choices(ideal, distractors, seed="fixed")
    assert result1 == result2


def test_build_choices_duplicate_ideal_and_distractor() -> None:
    """Correctly tracks correct answer even when ideal text matches a distractor.

    Rare but real: 1 of 600 SeqQA questions has the ideal duplicated in distractors.
    """
    ideal = "same"
    distractors = ["same", "other"]
    choices, correct_idx = build_choices(ideal, distractors, seed="test")
    assert len(choices) == 4  # ideal + 2 distractors + insufficient info
    assert choices[correct_idx] == ideal


def test_build_choices_different_seeds_can_differ() -> None:
    """Different seeds can produce different orderings."""
    ideal = "correct"
    distractors = ["a", "b", "c", "d", "e", "f"]
    results = {tuple(build_choices(ideal, distractors, seed=str(i))[0]) for i in range(20)}
    # With 8 items and 20 seeds, extremely unlikely all are the same ordering
    assert len(results) > 1


# --- Tests for format_question ---


def test_format_question_structure() -> None:
    """Formatted prompt matches the paper's CoT template."""
    prompt = format_question("What is DNA?", ["Protein", "Nucleic acid", "Sugar"])
    assert prompt.startswith("The following is a multiple choice question about biology.")
    assert "Think step by step." in prompt
    assert "Question: What is DNA?" in prompt
    assert "Options:" in prompt
    assert "(A) Protein" in prompt
    assert "(B) Nucleic acid" in prompt
    assert "(C) Sugar" in prompt
    assert "[ANSWER]" in prompt
    assert "[/ANSWER]" in prompt


@pytest.mark.parametrize(
    "num_choices",
    [2, 4, 7],
)
def test_format_question_variable_choice_count(num_choices: int) -> None:
    """Works with different numbers of choices."""
    choices = [f"option_{i}" for i in range(num_choices)]
    prompt = format_question("Q?", choices)
    for i in range(num_choices):
        letter = chr(ord("A") + i)
        assert f"({letter}) option_{i}" in prompt


def test_subtask_enum_values_match_dataset_configs() -> None:
    """Enum values match the HuggingFace dataset config names."""
    expected = {"LitQA2", "DbQA", "SuppQA", "ProtocolQA", "SeqQA", "CloningScenarios"}
    actual = {s.value for s in LabBenchSubtask}
    assert actual == expected


# --- Tests for extract_answer ---


def test_extract_answer_from_tags() -> None:
    """Primary extraction: [ANSWER]X[/ANSWER] tags."""
    assert extract_answer("Let me think... [ANSWER]B[/ANSWER]") == "B"
    assert extract_answer("[ANSWER] C [/ANSWER]") == "C"  # with whitespace


def test_extract_answer_from_answer_is_pattern() -> None:
    """Fallback 1: 'answer is X' pattern."""
    assert extract_answer("After analysis, the answer is (D).") == "D"
    assert extract_answer("the answer is B") == "B"


def test_extract_answer_last_letter_fallback() -> None:
    """Fallback 2: last isolated capital letter."""
    assert extract_answer("I think the correct option is C because...") == "C"


def test_extract_answer_returns_none_on_failure() -> None:
    """Returns None when no answer can be extracted."""
    assert extract_answer("no single capital letters here at all") is None
    assert extract_answer("") is None


def test_extract_answer_prefers_tags_over_fallbacks() -> None:
    """[ANSWER] tags take priority over 'answer is' and last-letter fallbacks."""
    text = "The answer is A but actually [ANSWER]C[/ANSWER]"
    assert extract_answer(text) == "C"


# --- Helper to construct a LabBenchEvaluation with mock questions ---


def make_evaluation(
    questions: list[LabBenchQuestion], subtasks: list[LabBenchSubtask] | None = None
) -> LabBenchEvaluation:
    """Create a LabBenchEvaluation with pre-loaded questions (no HF download)."""
    if subtasks is None:
        subtasks = list({q.subtask for q in questions})

    config = LabBenchEvaluationConfig(
        model_checkpoint="test-model",
        out_dir="/tmp/lab-bench-test",
        model_config=ModelConfig(
            template_name=TemplateName.PLAIN,
            max_generation_length=1,
            inference_batch_size=32,
        ),
        subtasks=subtasks,
    )
    evaluation = LabBenchEvaluation(config)
    # Inject questions directly into instance dict, bypassing HF dataset download.
    # This populates the cached_property cache so .questions returns our mock data.
    vars(evaluation)["questions"] = questions
    return evaluation


def _make_prompt_with_insufficient_info(question: str, choices: list[str], insufficient_info_letter: str) -> str:
    """Build a prompt where the 'Insufficient information' option is at a known letter."""
    # Place choices in order, with insufficient info at the specified position
    idx = ord(insufficient_info_letter) - ord("A")
    all_choices = list(choices)
    all_choices.insert(idx, INSUFFICIENT_INFO_OPTION)
    return format_question(question, all_choices)


def make_question(prompt: str, correct_answer: str, subtask: LabBenchSubtask, num_choices: int = 5) -> LabBenchQuestion:
    """Create a LabBenchQuestion for testing."""
    return LabBenchQuestion(
        prompt=prompt,
        correct_answer=correct_answer,
        subtask=subtask,
        num_choices=num_choices,
    )


# --- Tests for compute_scores ---


def test_compute_scores_all_correct() -> None:
    """All correct answers produce scores of 1.0."""
    questions = [
        make_question("Q1?", "A", LabBenchSubtask.LITQA2),
        make_question("Q2?", "B", LabBenchSubtask.DBQA),
        make_question("Q3?", "C", LabBenchSubtask.SEQQA),
    ]
    evaluation = make_evaluation(questions)

    inferences = InferenceSchema.validate(
        pl.DataFrame(
            {
                InferenceSchema.prompt: ["Q1?", "Q2?", "Q3?"],
                InferenceSchema.response: ["A", "B", "C"],
            }
        )
    )

    scores_df = evaluation.compute_scores(inferences)
    scores = scores_df["score"].to_list()
    assert scores == [1.0, 1.0, 1.0]


def test_compute_scores_all_wrong() -> None:
    """All wrong answers produce scores of 0.0."""
    questions = [
        make_question("Q1?", "A", LabBenchSubtask.LITQA2),
        make_question("Q2?", "B", LabBenchSubtask.DBQA),
    ]
    evaluation = make_evaluation(questions)

    inferences = InferenceSchema.validate(
        pl.DataFrame(
            {
                InferenceSchema.prompt: ["Q1?", "Q2?"],
                InferenceSchema.response: ["C", "D"],
            }
        )
    )

    scores_df = evaluation.compute_scores(inferences)
    scores = scores_df["score"].to_list()
    assert scores == [0.0, 0.0]


def test_compute_scores_mixed() -> None:
    """Mix of correct and wrong answers."""
    questions = [
        make_question("Q1?", "A", LabBenchSubtask.LITQA2),
        make_question("Q2?", "B", LabBenchSubtask.LITQA2),
        make_question("Q3?", "C", LabBenchSubtask.LITQA2),
    ]
    evaluation = make_evaluation(questions)

    inferences = InferenceSchema.validate(
        pl.DataFrame(
            {
                InferenceSchema.prompt: ["Q1?", "Q2?", "Q3?"],
                InferenceSchema.response: ["A", "D", "C"],
            }
        )
    )

    scores_df = evaluation.compute_scores(inferences)
    scores = scores_df["score"].to_list()
    assert scores == [1.0, 0.0, 1.0]


# --- Tests for compute_results ---


def test_compute_results_single_subtask() -> None:
    """Accuracy, precision, and coverage computed for a single subtask and overall."""
    questions = [
        make_question("Q1?", "A", LabBenchSubtask.LITQA2),
        make_question("Q2?", "B", LabBenchSubtask.LITQA2),
    ]
    evaluation = make_evaluation(questions, subtasks=[LabBenchSubtask.LITQA2])

    scores_df = ScoreSchema.validate(
        pl.DataFrame(
            {
                ScoreSchema.prompt: ["Q1?", "Q2?"],
                ScoreSchema.response: ["A", "D"],
                ScoreSchema.score: [1.0, 0.0],
            }
        )
    )

    results_df = evaluation.compute_results(scores_df)
    metrics = dict(
        zip(
            results_df[EvaluationSchema.metric_name].to_list(),
            results_df["metric_value"].to_list(),
            strict=True,
        )
    )

    assert metrics[str(MetricName.LAB_BENCH_LITQA2_ACCURACY)] == pytest.approx(0.5)
    assert metrics[str(MetricName.LAB_BENCH_ACCURACY)] == pytest.approx(0.5)
    # Both responses are non-insufficient, so coverage=1.0 and precision=accuracy
    assert metrics[str(MetricName.LAB_BENCH_COVERAGE)] == pytest.approx(1.0)
    assert metrics[str(MetricName.LAB_BENCH_PRECISION)] == pytest.approx(0.5)


def test_compute_results_multiple_subtasks() -> None:
    """Per-subtask and overall accuracy computed correctly across subtasks."""
    questions = [
        make_question("Q1?", "A", LabBenchSubtask.LITQA2),
        make_question("Q2?", "B", LabBenchSubtask.LITQA2),
        make_question("Q3?", "C", LabBenchSubtask.DBQA),
        make_question("Q4?", "D", LabBenchSubtask.DBQA),
        make_question("Q5?", "A", LabBenchSubtask.DBQA),
    ]
    evaluation = make_evaluation(questions, subtasks=[LabBenchSubtask.LITQA2, LabBenchSubtask.DBQA])

    # LitQA2: 1/2 correct, DbQA: 2/3 correct, overall: 3/5
    scores_df = ScoreSchema.validate(
        pl.DataFrame(
            {
                ScoreSchema.prompt: ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
                ScoreSchema.response: ["A", "X", "C", "D", "B"],
                ScoreSchema.score: [1.0, 0.0, 1.0, 1.0, 0.0],
            }
        )
    )

    results_df = evaluation.compute_results(scores_df)
    metrics = dict(
        zip(
            results_df[EvaluationSchema.metric_name].to_list(),
            results_df["metric_value"].to_list(),
            strict=True,
        )
    )

    assert metrics[str(MetricName.LAB_BENCH_LITQA2_ACCURACY)] == pytest.approx(0.5)
    assert metrics[str(MetricName.LAB_BENCH_DBQA_ACCURACY)] == pytest.approx(2 / 3)
    assert metrics[str(MetricName.LAB_BENCH_ACCURACY)] == pytest.approx(3 / 5)


def test_compute_results_perfect_score() -> None:
    """All correct produces 1.0 across the board."""
    questions = [
        make_question("Q1?", "A", LabBenchSubtask.SEQQA),
        make_question("Q2?", "B", LabBenchSubtask.SEQQA),
    ]
    evaluation = make_evaluation(questions, subtasks=[LabBenchSubtask.SEQQA])

    scores_df = ScoreSchema.validate(
        pl.DataFrame(
            {
                ScoreSchema.prompt: ["Q1?", "Q2?"],
                ScoreSchema.response: ["A", "B"],
                ScoreSchema.score: [1.0, 1.0],
            }
        )
    )

    results_df = evaluation.compute_results(scores_df)
    metrics = dict(
        zip(
            results_df[EvaluationSchema.metric_name].to_list(),
            results_df["metric_value"].to_list(),
            strict=True,
        )
    )

    assert metrics[str(MetricName.LAB_BENCH_SEQQA_ACCURACY)] == pytest.approx(1.0)
    assert metrics[str(MetricName.LAB_BENCH_ACCURACY)] == pytest.approx(1.0)


def test_compute_results_precision_and_coverage_with_insufficient_info() -> None:
    """Precision and coverage correctly account for 'Insufficient information' responses."""
    # Build prompts where we know which letter is "Insufficient information"
    p1 = _make_prompt_with_insufficient_info("What enzyme?", ["Lipase", "Protease", "Kinase", "Amylase"], "E")
    p2 = _make_prompt_with_insufficient_info("What molecule?", ["ATP", "DNA", "RNA", "GTP"], "C")

    questions = [
        make_question(p1, "A", LabBenchSubtask.LITQA2, num_choices=5),
        make_question(p2, "A", LabBenchSubtask.LITQA2, num_choices=5),
    ]
    evaluation = make_evaluation(questions, subtasks=[LabBenchSubtask.LITQA2])

    # Q1: model answers "A" (correct, attempted)
    # Q2: model answers "C" (which is "Insufficient information", not attempted)
    scores_df = ScoreSchema.validate(
        pl.DataFrame(
            {
                ScoreSchema.prompt: [p1, p2],
                ScoreSchema.response: ["A", "C"],
                ScoreSchema.score: [1.0, 0.0],
            }
        )
    )

    results_df = evaluation.compute_results(scores_df)
    metrics = dict(
        zip(
            results_df[EvaluationSchema.metric_name].to_list(),
            results_df["metric_value"].to_list(),
            strict=True,
        )
    )

    # Accuracy: 1/2 = 0.5
    assert metrics[str(MetricName.LAB_BENCH_ACCURACY)] == pytest.approx(0.5)
    # Coverage: 1/2 = 0.5 (only Q1 was attempted)
    assert metrics[str(MetricName.LAB_BENCH_COVERAGE)] == pytest.approx(0.5)
    # Precision: 1/1 = 1.0 (Q1 was attempted and correct)
    assert metrics[str(MetricName.LAB_BENCH_PRECISION)] == pytest.approx(1.0)
