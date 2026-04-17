"""LLM-as-a-judge for MMLU-Pro response grading.

Synchronous judge that sees the original question, choices, and ground-truth
answer to determine whether a model response is correct. Unlike the regex
parser, it can grade responses that didn't follow the expected format of
indicating the letter of the answer.
"""

import logging
import random
import time

from openai import APIStatusError, APITimeoutError, OpenAI, RateLimitError

MAX_RETRIES = 5
RETRY_DELAY_SECS = 1

SYSTEM_PROMPT = """You are grading a model's response to a multiple-choice question.

You will be given:
1. The original question and answer choices
2. The correct answer letter
3. The model's response

Your task is to determine if the model's response clearly indicates the CORRECT answer.

Return ONLY one of:
- "correct" - if the model's response clearly and unambiguously indicates the correct answer
- "incorrect" - if the model's response clearly indicates a wrong answer or you cannot determine what answer the model intended"""


def format_grading_prompt(
    question: str,
    choices: str,
    ground_truth_answer: str,
    model_response: str,
) -> str:
    """Format the prompt for the grading judge."""
    return f"""## Question
{question}

## Answer Choices
{choices}

## Correct Answer
{ground_truth_answer}

## Model's Response
{model_response}"""


def grade_with_context(
    client: OpenAI,
    model: str,
    question: str,
    choices: str,
    ground_truth_answer: str,
    model_response: str,
) -> float:
    """Grade a model response with full question context.

    Returns 1.0 for correct, 0.0 for incorrect.
    """
    if not model_response.strip():
        return 0.0

    prompt = format_grading_prompt(question, choices, ground_truth_answer, model_response)
    # This substring check is fragile, but we expect any unlisted reasoning
    # models to throw an error since the non-reasoning codepath sets
    # temperature=0, which so far all OpenAI reasoning models disallow.
    is_reasoning_model = any(tag in model for tag in ("gpt-5", "o1", "o3"))

    attempt = 0
    while True:
        attempt += 1
        try:
            if is_reasoning_model:
                # We don't want to set max_tokens since setting it too low can
                # mean we run out of tokens during the reasoning phase.
                # Temperature is also not configurable for reasoning models.
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    reasoning_effort="low",
                )
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=10,
                    temperature=0,
                )
            content = resp.choices[0].message.content
            if content is None:
                logging.warning("Judge returned None content (attempt %d/%d)", attempt, MAX_RETRIES)
                if attempt >= MAX_RETRIES:
                    return 0.0
                time.sleep(RETRY_DELAY_SECS)
                continue

            result = content.strip().lower()
            if "correct" in result and "incorrect" not in result:
                return 1.0
            elif "incorrect" in result:
                return 0.0
            else:
                logging.warning("Unexpected judge response (attempt %d/%d): %r", attempt, MAX_RETRIES, result)
                if attempt >= MAX_RETRIES:
                    return 0.0
                time.sleep(RETRY_DELAY_SECS)

        except APITimeoutError as e:
            if attempt >= MAX_RETRIES:
                raise Exception(f"Max retries ({MAX_RETRIES}) exceeded due to timeout") from e
            time.sleep(RETRY_DELAY_SECS + random.uniform(0, 0.5))

        except RateLimitError as e:
            if attempt >= MAX_RETRIES:
                raise Exception(f"Max retries ({MAX_RETRIES}) exceeded due to rate limiting") from e
            time.sleep(RETRY_DELAY_SECS + random.uniform(0.5, 1.5))

        except APIStatusError as e:
            if 500 <= e.status_code < 600 and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECS + random.uniform(0, 0.5))
            else:
                raise Exception(f"API error (status {e.status_code}) after {attempt} attempts") from e
