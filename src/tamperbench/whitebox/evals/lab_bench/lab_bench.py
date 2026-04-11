"""LAB-Bench (Language Agent Biology Benchmark) Evaluation.

Measures biology research capabilities via multiple-choice questions across
text-based subtasks: literature reasoning, database navigation, protocol
troubleshooting, sequence manipulation, and molecular cloning.

Reference:
    - Paper: "LAB-Bench: Measuring Capabilities of Language Models for Biology Research"
             https://arxiv.org/abs/2407.10362
    - Dataset: https://huggingface.co/datasets/futurehouse/lab-bench
"""

import logging
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from itertools import chain
from typing import Any, cast

import polars as pl
import torch
from datasets import load_dataset
from pandera.typing.polars import DataFrame
from typing_extensions import override

from tamperbench.whitebox.evals.base import WhiteBoxEvaluation, WhiteBoxEvaluationConfig, load_vllm_model_and_tokenizer
from tamperbench.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from tamperbench.whitebox.evals.reference import ReferenceScore, ReferenceScores
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.evals.utils import format_chat_prompt, generate
from tamperbench.whitebox.utils import EvalName, MetricName, OptimizationDirection

logger = logging.getLogger(__name__)

DATASET_ID = "futurehouse/lab-bench"

# The "Insufficient information" option that the paper adds to every question.
INSUFFICIENT_INFO_OPTION = "Insufficient information to answer the question"

# Note: The HF dataset is only the public split, there is another 20% of
# questions that are not publicly available.

# Reference: Measured empirically with this implementation.
# The LAB-Bench paper (Ribeiro et al., 2024, Table 2) reports results for
# GPT-4, Claude 3 Opus, Gemini 1.5 Pro, and Llama-3-70B-Instruct, but not
# for 8B-class models.
LAB_BENCH_REFERENCE_SCORES = ReferenceScores(
    scores=[
        ReferenceScore(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            metrics={
                str(MetricName.LAB_BENCH_ACCURACY): 0.2763,
                str(MetricName.LAB_BENCH_LITQA2_ACCURACY): 0.2714,
                str(MetricName.LAB_BENCH_DBQA_ACCURACY): 0.2096,
                str(MetricName.LAB_BENCH_SUPPQA_ACCURACY): 0.2683,
                str(MetricName.LAB_BENCH_PROTOCOLQA_ACCURACY): 0.4167,
                str(MetricName.LAB_BENCH_SEQQA_ACCURACY): 0.3133,
                str(MetricName.LAB_BENCH_CLONING_ACCURACY): 0.2424,
            },
            source="Empirical measurement (TamperBench, April 2026)",
            tolerance=0.05,
        ),
    ]
)


class LabBenchSubtask(str, Enum):
    """LAB-Bench subtask configs (text-only)."""

    LITQA2 = "LitQA2"
    DBQA = "DbQA"
    SUPPQA = "SuppQA"
    PROTOCOLQA = "ProtocolQA"
    SEQQA = "SeqQA"
    CLONING = "CloningScenarios"


SUBTASK_METRIC: dict[LabBenchSubtask, MetricName] = {
    LabBenchSubtask.LITQA2: MetricName.LAB_BENCH_LITQA2_ACCURACY,
    LabBenchSubtask.DBQA: MetricName.LAB_BENCH_DBQA_ACCURACY,
    LabBenchSubtask.SUPPQA: MetricName.LAB_BENCH_SUPPQA_ACCURACY,
    LabBenchSubtask.PROTOCOLQA: MetricName.LAB_BENCH_PROTOCOLQA_ACCURACY,
    LabBenchSubtask.SEQQA: MetricName.LAB_BENCH_SEQQA_ACCURACY,
    LabBenchSubtask.CLONING: MetricName.LAB_BENCH_CLONING_ACCURACY,
}

SUBTASK_PRECISION_METRIC: dict[LabBenchSubtask, MetricName] = {
    LabBenchSubtask.LITQA2: MetricName.LAB_BENCH_LITQA2_PRECISION,
    LabBenchSubtask.DBQA: MetricName.LAB_BENCH_DBQA_PRECISION,
    LabBenchSubtask.SUPPQA: MetricName.LAB_BENCH_SUPPQA_PRECISION,
    LabBenchSubtask.PROTOCOLQA: MetricName.LAB_BENCH_PROTOCOLQA_PRECISION,
    LabBenchSubtask.SEQQA: MetricName.LAB_BENCH_SEQQA_PRECISION,
    LabBenchSubtask.CLONING: MetricName.LAB_BENCH_CLONING_PRECISION,
}

SUBTASK_COVERAGE_METRIC: dict[LabBenchSubtask, MetricName] = {
    LabBenchSubtask.LITQA2: MetricName.LAB_BENCH_LITQA2_COVERAGE,
    LabBenchSubtask.DBQA: MetricName.LAB_BENCH_DBQA_COVERAGE,
    LabBenchSubtask.SUPPQA: MetricName.LAB_BENCH_SUPPQA_COVERAGE,
    LabBenchSubtask.PROTOCOLQA: MetricName.LAB_BENCH_PROTOCOLQA_COVERAGE,
    LabBenchSubtask.SEQQA: MetricName.LAB_BENCH_SEQQA_COVERAGE,
    LabBenchSubtask.CLONING: MetricName.LAB_BENCH_CLONING_COVERAGE,
}


@dataclass(frozen=True)
class LabBenchQuestion:
    """Single LAB-Bench question with metadata."""

    prompt: str
    correct_answer: str
    subtask: LabBenchSubtask
    num_choices: int


@dataclass
class LabBenchEvaluationConfig(WhiteBoxEvaluationConfig):
    """Configuration for LAB-Bench evaluation."""

    subtasks: list[LabBenchSubtask] = field(
        default_factory=lambda: list(LabBenchSubtask),
    )


def build_choices(ideal: str, distractors: list[str], seed: str) -> tuple[list[str], int]:
    """Combine ideal answer with distractors, add "Insufficient information", and shuffle.

    Returns (shuffled_choices, index_of_correct_answer).
    """
    all_options = [ideal, *distractors, INSUFFICIENT_INFO_OPTION]
    indexed = list(enumerate(all_options))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    choices = [text for _, text in indexed]
    correct_idx = next(i for i, (orig_i, _) in enumerate(indexed) if orig_i == 0)
    return choices, correct_idx


def format_question(question_text: str, choices: list[str]) -> str:
    """Format a LAB-Bench question using the paper's zero-shot CoT prompt template.

    Matches the prompt from LAB-Bench paper Section 3:
    chain-of-thought with [ANSWER]/[/ANSWER] tags for automatic parsing.
    """
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    options_text = "\n".join(f"({letter}) {choice}" for letter, choice in zip(letters, choices, strict=True))
    return (
        f"The following is a multiple choice question about biology.\n"
        f"Please answer by responding with the letter of the correct answer.\n\n"
        f"Think step by step.\n\n"
        f"Question: {question_text}\n\n"
        f"Options:\n"
        f"{options_text}\n\n"
        f"You MUST include the letter of the correct answer within the following tags: "
        f"[ANSWER] and [/ANSWER]. For example, '[ANSWER]<answer>[/ANSWER]', "
        f"where <answer> is the correct letter. Always answer in exactly this format "
        f"of a single letter between the two tags, even if you are unsure. "
        f"We require this because we use automatic parsing."
    )


def extract_answer(text: str) -> str | None:
    """Extract answer letter from model response using [ANSWER] tags, with fallbacks.

    Extraction cascade:
    1. [ANSWER]X[/ANSWER] tags (paper's prescribed format)
    2. "answer is (X)" pattern (common CoT output)
    3. Last isolated capital letter A-J (last resort)
    """
    # Primary: [ANSWER]X[/ANSWER] tags
    match = re.search(r"\[ANSWER\]\s*([A-J])\s*\[/ANSWER\]", text)
    if match:
        return match.group(1)

    # Fallback 1: "answer is (X)" or "answer is X"
    match = re.search(r"answer is \(?([A-J])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback 2: last isolated capital letter
    match = re.search(r"\b([A-J])\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    if match:
        return match.group(1)

    return None


@register_evaluation(EvalName.LAB_BENCH)
class LabBenchEvaluation(WhiteBoxEvaluation[LabBenchEvaluationConfig]):
    """LAB-Bench Evaluation using multiple-choice questions.

    Measures model capability on biology research tasks.
    Higher accuracy = better capability.

    For defenders: Higher accuracy is better (MAXIMIZE).
    For attackers: Lower accuracy is better (MINIMIZE).

    Direction rationale: LAB-Bench is a **benign capability** eval, not a safety eval.
    Although it covers biology, its subtasks measure general research skills (literature
    comprehension, database querying, protocol understanding, sequence manipulation) rather
    than dual-use dangerous knowledge. WMDP-Bio specifically targets hazardous biosecurity
    knowledge (e.g., synthesizing pathogens) and is treated as a safety eval. Here, the
    defender wants the model to *retain* biology research competence after safety training,
    so defender=MAXIMIZE. The attacker tries to *degrade* this competence, so
    attacker=MINIMIZE.
    """

    name: EvalName = EvalName.LAB_BENCH
    objective: MetricName = MetricName.LAB_BENCH_ACCURACY
    attacker_direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    reference_scores: ReferenceScores = LAB_BENCH_REFERENCE_SCORES

    def __init__(self, eval_config: LabBenchEvaluationConfig) -> None:
        """Initialize LAB-Bench evaluation."""
        super().__init__(eval_config)

    @cached_property
    def questions(self) -> list[LabBenchQuestion]:
        """Load all LAB-Bench questions (cached on first access)."""
        return list(chain.from_iterable(self._load_subtask_questions(subtask) for subtask in self.eval_config.subtasks))

    def _load_subtask_questions(self, subtask: LabBenchSubtask) -> list[LabBenchQuestion]:
        """Load questions for a single subtask."""
        # LAB-Bench has only a "train" split (no test/val split in the HF dataset)
        dataset = load_dataset(DATASET_ID, subtask.value, split="train")

        questions = []
        for raw_item in dataset:
            item = cast(dict[str, Any], raw_item)
            assert "id" in item, f"LAB-Bench item missing 'id' field in {subtask.value}"
            seed = str(item["id"])
            choices, correct_idx = build_choices(item["ideal"], item["distractors"], seed)
            answer_letter = chr(ord("A") + correct_idx)

            # Match per-subtask question formatting from the original LAB-Bench
            # implementations (Future-House/LAB-Bench {Subtask}/task.py):
            # - SuppQA: prepend paper title and DOI
            # - ProtocolQA: prepend protocol text
            question_text = item["question"]
            if subtask == LabBenchSubtask.SUPPQA:
                question_text = f"Paper title: {item['paper-title']}\nDOI: {item['source']}\n{question_text}"
            elif subtask == LabBenchSubtask.PROTOCOLQA:
                question_text = item["protocol"] + question_text

            questions.append(
                LabBenchQuestion(
                    prompt=format_question(question_text, choices),
                    correct_answer=answer_letter,
                    subtask=subtask,
                    num_choices=len(choices),
                )
            )
        return questions

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run model on LAB-Bench questions using CoT generation + regex extraction.

        Following the paper's protocol: generate a chain-of-thought response, then
        extract the answer letter from [ANSWER]X[/ANSWER] tags via regex.
        """
        model, tokenizer = load_vllm_model_and_tokenizer(self.eval_config)

        raw_prompts = [q.prompt for q in self.questions]
        # Wrap each question in the model's chat template (user turn) so instruct
        # models get the proper framing for CoT generation.
        chat_prompts = [format_chat_prompt(p, tokenizer) for p in raw_prompts]

        max_new_tokens = min(self.eval_config.model_config.max_generation_length, 2048)
        responses = generate(chat_prompts, model, max_new_tokens=max_new_tokens, temperature=0.0)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract answer letters from CoT responses.
        # If extraction fails, store "?" so it scores as wrong (won't match any
        # answer letter) and counts as attempted (not the insufficient-info letter).
        answers: list[str] = []
        for i, response in enumerate(responses):
            extracted = extract_answer(response)
            if extracted is None:
                logger.warning(
                    "LAB-Bench answer extraction failed for question %d. Response: %s",
                    i,
                    response[:200],
                )
                extracted = "?"
            answers.append(extracted)

        # Store raw (pre-chat-template) prompts as join key for compute_scores.
        return InferenceSchema.validate(
            pl.DataFrame(
                {
                    InferenceSchema.prompt: raw_prompts,
                    InferenceSchema.response: answers,
                }
            )
        )

    @cached_property
    def _insufficient_info_letters(self) -> dict[str, str]:
        """Map each prompt to the letter assigned to its "Insufficient information" option."""
        result = {}
        for q in self.questions:
            # Find the letter for "Insufficient information" in this question's prompt.
            # Options are formatted as "(X) choice_text".
            lines = q.prompt.split("\n")
            for line in lines:
                stripped = line.strip()
                if len(stripped) >= 4 and stripped[0] == "(" and stripped[2] == ")" and stripped[1].isalpha():
                    letter = stripped[1]
                    option_text = stripped[4:]  # skip "(X) "
                    if option_text == INSUFFICIENT_INFO_OPTION:
                        result[q.prompt] = letter
                        break
        return result

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Compute accuracy scores (0 or 1) for each question."""
        # NOTE: Uses full prompt string as join key between questions and inferences.
        # This is fragile if any transformation (serialization, whitespace normalization)
        # modifies prompts between compute_inferences and compute_scores. Currently safe
        # because both methods run in the same process without serialization round-trips.
        prompt_to_answer = {q.prompt: q.correct_answer for q in self.questions}
        prompts = inferences[InferenceSchema.prompt].to_list()
        responses = inferences[InferenceSchema.response].to_list()

        scores = [
            float(response == prompt_to_answer[prompt]) for prompt, response in zip(prompts, responses, strict=True)
        ]

        return ScoreSchema.validate(
            pl.DataFrame(
                {
                    ScoreSchema.prompt: prompts,
                    ScoreSchema.response: responses,
                    ScoreSchema.score: scores,
                }
            )
        )

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        """Compute accuracy, precision, and coverage metrics per subtask and overall.

        Following the paper (Ribeiro et al., 2024):
        - Accuracy: correct / total
        - Precision (selective accuracy): correct / attempted (excluding "Insufficient info")
        - Coverage: attempted / total
        """
        prompt_to_subtask = {q.prompt: q.subtask for q in self.questions}
        insuf_letters = self._insufficient_info_letters
        scores_list = scores["score"].to_list()
        prompts = scores[ScoreSchema.prompt].to_list()
        responses = scores[ScoreSchema.response].to_list()

        subtask_scores: dict[LabBenchSubtask, list[float]] = {s: [] for s in self.eval_config.subtasks}
        subtask_attempted: dict[LabBenchSubtask, list[float]] = {s: [] for s in self.eval_config.subtasks}
        all_attempted: list[float] = []

        for prompt, response, score in zip(prompts, responses, scores_list, strict=True):
            subtask = prompt_to_subtask.get(prompt)
            attempted = response != insuf_letters.get(prompt)
            if subtask is not None:
                subtask_scores[subtask].append(score)
                subtask_attempted[subtask].append(float(attempted))
            all_attempted.append(float(attempted))

        metrics: list[tuple[str, float]] = []
        for subtask in self.eval_config.subtasks:
            if not subtask_scores[subtask]:
                continue
            n = len(subtask_scores[subtask])
            correct = sum(subtask_scores[subtask])
            attempted = sum(subtask_attempted[subtask])
            metrics.append((str(SUBTASK_METRIC[subtask]), correct / n))
            metrics.append((str(SUBTASK_PRECISION_METRIC[subtask]), correct / attempted if attempted > 0 else 0.0))
            metrics.append((str(SUBTASK_COVERAGE_METRIC[subtask]), attempted / n))

        # Overall metrics are micro-averaged (weighted by subtask size).
        # SeqQA has ~600 questions vs CloningScenarios ~33, so subtask sizes
        # vary by ~18x. Per-subtask metrics above allow macro-style analysis.
        total = len(scores_list)
        total_correct = sum(scores_list)
        total_attempted = sum(all_attempted)
        metrics.append((str(MetricName.LAB_BENCH_ACCURACY), total_correct / total if total else 0.0))
        metrics.append(
            (str(MetricName.LAB_BENCH_PRECISION), total_correct / total_attempted if total_attempted > 0 else 0.0)
        )
        metrics.append((str(MetricName.LAB_BENCH_COVERAGE), total_attempted / total if total else 0.0))

        return EvaluationSchema.validate(
            pl.DataFrame(
                {
                    EvaluationSchema.metric_name: [m[0] for m in metrics],
                    EvaluationSchema.metric_value: [m[1] for m in metrics],
                }
            )
        )
