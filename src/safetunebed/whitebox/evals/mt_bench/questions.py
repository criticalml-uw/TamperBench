"""MT-Bench questions loader.

Questions sourced from: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf
"""

from dataclasses import dataclass

from safetunebed.whitebox.utils import get_dataset

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


def load_mt_bench_questions(force_download: bool = False) -> list[MTBenchQuestion]:
    """Load MT-Bench questions from local cache or download if needed.

    On first run, downloads the questions from the FastChat repository and caches
    them locally in the data/ directory. Subsequent runs load from the local file.

    Args:
        force_download: If True, re-download the questions even if cached locally.

    Returns:
        list[MTBenchQuestion]: List of 80 MT-Bench questions.
    """
    data = get_dataset(
        MT_BENCH_QUESTIONS_URL, data_format="jsonl", force_download=force_download
    )
    return [
        MTBenchQuestion(
            question_id=item["question_id"],
            category=item["category"],
            turns=item["turns"],
            reference=item.get("reference"),
        )
        for item in data
    ]


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
