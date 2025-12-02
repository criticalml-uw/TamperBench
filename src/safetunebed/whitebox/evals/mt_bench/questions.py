"""MT-Bench questions loader.

Questions sourced from: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf
"""

import json
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

MT_BENCH_QUESTIONS_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"

CATEGORIES = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]


@dataclass
class MTBenchQuestion:
    """A single MT-Bench question with two turns."""

    question_id: int
    category: str
    turns: list[str]  # Always 2 turns
    reference: list[str] | None = None  # Reference answers if available


@lru_cache(maxsize=1)
def load_mt_bench_questions() -> list[MTBenchQuestion]:
    """Load MT-Bench questions from the FastChat repository.

    Returns:
        list[MTBenchQuestion]: List of 80 MT-Bench questions.
    """
    with urllib.request.urlopen(MT_BENCH_QUESTIONS_URL) as response:
        lines = response.read().decode("utf-8").strip().split("\n")

    questions = []
    for line in lines:
        data = json.loads(line)
        questions.append(
            MTBenchQuestion(
                question_id=data["question_id"],
                category=data["category"],
                turns=data["turns"],
                reference=data.get("reference"),
            )
        )

    return questions


def load_mt_bench_questions_by_category(
    categories: list[str] | None = None,
) -> list[MTBenchQuestion]:
    """Load MT-Bench questions filtered by category.

    Args:
        categories: List of categories to include. If None, include all.

    Returns:
        list[MTBenchQuestion]: Filtered list of questions.
    """
    questions = load_mt_bench_questions()
    if categories is None:
        return questions
    return [q for q in questions if q.category in categories]
