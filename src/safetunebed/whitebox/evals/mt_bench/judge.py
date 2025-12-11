"""LLM-as-a-Judge implementation for MT-Bench scoring.

Prompts sourced verbatim from:
https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl

Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf
"""

import re

from openai import OpenAI

# ============================================================================
# Single-answer grading prompts (from FastChat judge_prompts.jsonl)
# ============================================================================

# single-v1: General single-turn
SINGLE_V1_SYSTEM = "You are a helpful assistant."
SINGLE_V1_TEMPLATE = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

# single-math-v1: Math/coding single-turn with reference
SINGLE_MATH_V1_SYSTEM = "You are a helpful assistant."
SINGLE_MATH_V1_TEMPLATE = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Reference Answer]
{ref_answer_1}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

# single-v1-multi-turn: General multi-turn
SINGLE_V1_MULTI_TURN_SYSTEM = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

"""
SINGLE_V1_MULTI_TURN_TEMPLATE = """<|The Start of Assistant A's Conversation with User|>

### User:
{question_1}

### Assistant A:
{answer_1}

### User:
{question_2}

### Assistant A:
{answer_2}

<|The End of Assistant A's Conversation with User|>"""

# single-math-v1-multi-turn: Math/coding multi-turn with reference
SINGLE_MATH_V1_MULTI_TURN_SYSTEM = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. You evaluation should focus on the assistant's answer to the second question. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

"""
SINGLE_MATH_V1_MULTI_TURN_TEMPLATE = """<|The Start of Reference Answer|>

### User:
{question_1}

### Reference answer:
{ref_answer_1}

### User:
{question_2}

### Reference answer:
{ref_answer_2}

<|The End of Reference Answer|>


<|The Start of Assistant A's Conversation with User|>

### User:
{question_1}

### Assistant A:
{answer_1}

### User:
{question_2}

### Assistant A:
{answer_2}

<|The End of Assistant A's Conversation with User|>"""


def extract_rating(judge_response: str) -> float:
    """Extract numerical rating from judge response.

    Args:
        judge_response: The full response from the judge model.

    Returns:
        float: The extracted rating (1-10), or 0.0 if extraction fails.
    """
    # Try to find [[rating]] pattern (the official format)
    match = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", judge_response)
    if match:
        return float(match.group(1))

    # Fallback: look for "Rating: X" pattern
    match = re.search(r"[Rr]ating:\s*(\d+(?:\.\d+)?)", judge_response)
    if match:
        return float(match.group(1))

    print(f"No rating found in judge response: {judge_response}")
    return 0.0


def call_judge(
    client: OpenAI,
    judge_model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Make a call to the judge model.

    Args:
        client: OpenAI client.
        judge_model: Model to use for judging.
        system_prompt: System message for the judge.
        user_prompt: User message with the content to judge.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens for response.

    Returns:
        str: The judge's response.
    """
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def score_single_turn(
    client: OpenAI,
    judge_model: str,
    question: str,
    answer: str,
    reference: str | None = None,
    category: str = "general",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> tuple[float, str]:
    """Score a single-turn response.

    Args:
        client: OpenAI client.
        judge_model: Model to use for judging.
        question: The user question.
        answer: The assistant's answer.
        reference: Reference answer (for math/coding).
        category: Question category.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens for response.

    Returns:
        tuple[float, str]: (score, judge_explanation)
    """
    if reference and category in ("math", "coding"):
        system_prompt = SINGLE_MATH_V1_SYSTEM
        user_prompt = SINGLE_MATH_V1_TEMPLATE.format(
            question=question, answer=answer, ref_answer_1=reference
        )
    else:
        system_prompt = SINGLE_V1_SYSTEM
        user_prompt = SINGLE_V1_TEMPLATE.format(question=question, answer=answer)

    judge_response = call_judge(
        client, judge_model, system_prompt, user_prompt, temperature, max_tokens
    )
    return extract_rating(judge_response), judge_response


def score_multi_turn(
    client: OpenAI,
    judge_model: str,
    question_1: str,
    answer_1: str,
    question_2: str,
    answer_2: str,
    reference_1: str | None = None,
    reference_2: str | None = None,
    category: str = "general",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> tuple[float, str]:
    """Score a multi-turn response (focuses on second turn).

    Args:
        client: OpenAI client.
        judge_model: Model to use for judging.
        question_1: First user question.
        answer_1: First assistant answer.
        question_2: Second user question (follow-up).
        answer_2: Second assistant answer.
        reference_1: Reference for first turn (for math/coding).
        reference_2: Reference for second turn (for math/coding).
        category: Question category.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens for response.

    Returns:
        tuple[float, str]: (score, judge_explanation)
    """
    if reference_1 and reference_2 and category in ("math", "coding"):
        system_prompt = SINGLE_MATH_V1_MULTI_TURN_SYSTEM
        user_prompt = SINGLE_MATH_V1_MULTI_TURN_TEMPLATE.format(
            question_1=question_1,
            answer_1=answer_1,
            question_2=question_2,
            answer_2=answer_2,
            ref_answer_1=reference_1,
            ref_answer_2=reference_2,
        )
    else:
        system_prompt = SINGLE_V1_MULTI_TURN_SYSTEM
        user_prompt = SINGLE_V1_MULTI_TURN_TEMPLATE.format(
            question_1=question_1,
            answer_1=answer_1,
            question_2=question_2,
            answer_2=answer_2,
        )

    judge_response = call_judge(
        client, judge_model, system_prompt, user_prompt, temperature, max_tokens
    )
    return extract_rating(judge_response), judge_response
