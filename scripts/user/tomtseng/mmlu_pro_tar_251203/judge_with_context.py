"""LLM judge that grades MMLU-Pro responses with full question context.

This judge sees the original question, choices, and ground-truth answer,
allowing it to grade responses that didn't follow the expected format of
indicating the letter of the answer.

Much of the logic matches that of ../mmlu_pro_llm_judge_251105/judge.py.
"""

import asyncio
import logging
import random

import httpx
from dotenv import load_dotenv
from openai import APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

# ------------ config ------------
MODEL = "gpt-4.1"
HTTP_CONNECT_TIMEOUT = 5.0
HTTP_READ_TIMEOUT = 30.0  # longer timeout for more complex grading
HTTP_WRITE_TIMEOUT = 12.0
HTTP_POOL_TIMEOUT = 10.0
MAX_RETRIES = 5
RETRY_DELAY_SECS = 1
MAX_CONCURRENCY = 100  # lower concurrency since prompts are larger
# --------------------------------

limits = httpx.Limits(
    max_connections=128,
    max_keepalive_connections=64,
)
timeouts = httpx.Timeout(
    connect=HTTP_CONNECT_TIMEOUT,
    read=HTTP_READ_TIMEOUT,
    write=HTTP_WRITE_TIMEOUT,
    pool=HTTP_POOL_TIMEOUT,
)
http_client = httpx.AsyncClient(limits=limits, timeout=timeouts)

_default_client = AsyncOpenAI(
    http_client=http_client,
    max_retries=0,
)

SYSTEM_PROMPT = """You are grading a model's response to a multiple-choice question.

You will be given:
1. The original question and answer choices
2. The correct answer letter
3. The model's response

Your task is to determine if the model's response clearly indicates the CORRECT answer.

Return ONLY one of:
- "correct" - if the model's response clearly and unambiguously indicates the correct answer
- "incorrect" - if the model's response clearly indicates a wrong answer or you cannot determine what answer the model intended""".strip()

_default_sem = asyncio.Semaphore(MAX_CONCURRENCY)


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


async def _one_call(
    prompt: str,
    *,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    sem: asyncio.Semaphore | None = None,
) -> str:
    client = client or _default_client
    model = model or MODEL
    sem = sem or _default_sem
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    is_reasoning_model = any(tag in model for tag in ("gpt-5", "o1", "o3"))
    async with sem:
        if is_reasoning_model:
            # We don't want to set max_tokens since setting it too low can mean
            # we run out of tokens during the reasoning phase.
            # Temperature is also not configurable for reasoning models.
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort="low",
            )
        else:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=10,
                temperature=0,
            )
    content = resp.choices[0].message.content
    assert content is not None, f"OpenAI client returned None: {model=} {messages=} {resp=}"
    return content.strip().lower()


async def grade_with_context(
    question: str,
    choices: str,
    ground_truth_answer: str,
    model_response: str,
    *,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    sem: asyncio.Semaphore | None = None,
) -> str:
    """Grade a model response with full question context.

    Args:
        question: The original question text.
        choices: The answer choices formatted as a string.
        ground_truth_answer: The correct answer letter.
        model_response: The model's response to grade.
        client: OpenAI client to use. Defaults to module-level client.
        model: Model name to use. Defaults to module-level MODEL.
        sem: Semaphore to use for concurrency. Defaults to module-level semaphore.

    Returns: "correct" or "incorrect"
    """
    if not model_response.strip():
        # The answer is blank which is clearly wrong and can mess with the
        # judge.
        return "incorrect"

    prompt = format_grading_prompt(question, choices, ground_truth_answer, model_response)

    attempt = 0
    while True:
        attempt += 1
        try:
            result = await _one_call(prompt, client=client, model=model, sem=sem)
            # Normalize the response
            if "correct" in result and "incorrect" not in result:
                return "correct"
            elif "incorrect" in result:
                return "incorrect"
            else:
                logging.warning(
                    "Unexpected judge response (attempt %d/%d): %r\n  Prompt: %s",
                    attempt,
                    MAX_RETRIES,
                    result,
                    prompt,
                )
                if attempt >= MAX_RETRIES:
                    return "incorrect"

        except (APITimeoutError, asyncio.TimeoutError) as e:
            if attempt >= MAX_RETRIES:
                raise Exception(f"Max retries ({MAX_RETRIES}) exceeded due to timeout") from e
            await asyncio.sleep(RETRY_DELAY_SECS + random.uniform(0, 0.5))

        except RateLimitError as e:
            if attempt >= MAX_RETRIES:
                raise Exception(f"Max retries ({MAX_RETRIES}) exceeded due to rate limiting") from e
            await asyncio.sleep(RETRY_DELAY_SECS + random.uniform(0.5, 1.5))

        except APIStatusError as e:
            if 500 <= e.status_code < 600 and attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SECS + random.uniform(0, 0.5))
            else:
                raise Exception(f"API error (status {e.status_code}) after {attempt} attempts") from e

        except Exception as e:
            raise Exception(f"Unexpected error after {attempt} attempts: {e!r}") from e
