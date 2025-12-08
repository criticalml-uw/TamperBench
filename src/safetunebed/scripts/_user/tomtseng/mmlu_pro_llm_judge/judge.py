"""LLM judge for extract answers to MMLU-Pro.

Copied from
https://github.com/AlignmentResearch/scaling-poisoning-internal/pull/101/files
but with a modified system prompt.
"""

import asyncio
import random

import httpx
from dotenv import load_dotenv
from openai import APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

# Load environment variables from .env file
load_dotenv()

# ------------ config ------------
MODEL = "gpt-4.1"
HTTP_CONNECT_TIMEOUT = 5.0  # DNS/TCP connect
HTTP_READ_TIMEOUT = 12.0  # server response body
HTTP_WRITE_TIMEOUT = 12.0
HTTP_POOL_TIMEOUT = 10.0  # waiting for a free connection from pool
MAX_RETRIES = 5
RETRY_DELAY_SECS = 1
MAX_CONCURRENCY = 600
# --------------------------------

# Build an explicit httpx client with clear limits/timeouts.
limits = httpx.Limits(
    max_connections=128,  # total open sockets
    max_keepalive_connections=64,  # pooled keepalive sockets
)
timeouts = httpx.Timeout(
    connect=HTTP_CONNECT_TIMEOUT,
    read=HTTP_READ_TIMEOUT,
    write=HTTP_WRITE_TIMEOUT,
    pool=HTTP_POOL_TIMEOUT,
)
http_client = httpx.AsyncClient(limits=limits, timeout=timeouts)

client = AsyncOpenAI(
    http_client=http_client,
    max_retries=0,  # disable SDK internal retries; we control retries
)

SYSTEM_PROMPT = """You will be given an answer to a multiple-choice question.
The response should clearly contain a single-letter response. Return ONLY the letter (A, B, C, D, E, F, G, H, I, J) of the answer.

Look for patterns like "The answer is (X)" where X is a letter. Extract only the letter from the first such occurrence.

Return only the single capital letter with no explanation, no punctuation, no formatting.
If the response doesn't contain a single-letter answer, response with "n/a" instead.

If the response continues to generate additional multiple-choice questions and answers after the first answer, ignore those subsequent questions and answers, and extract ONLY the letter from the FIRST answer.""".strip()

sem = asyncio.Semaphore(MAX_CONCURRENCY)


async def _one_call(answer: str) -> str:
    async with sem:
        # IMPORTANT: do NOT pass `timeout=` here. Use the httpx client timeout instead.
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": answer},
            ],
            max_tokens=5,
            temperature=0,
        )
    return resp.choices[0].message.content.strip()


async def extract_with_retries(answer: str) -> str:
    """Extract single-letter answer from MMLU-Pro response.

    Returns "n/a" if no answer was found.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            # Let the HTTP client enforce the timeout; avoid double timeouts.
            return await _one_call(answer)

        except (APITimeoutError, asyncio.TimeoutError) as e:
            # network/SDK timeout
            if attempt >= MAX_RETRIES:
                raise Exception(
                    f"Max retries ({MAX_RETRIES}) exceeded due to timeout"
                ) from e
            await asyncio.sleep(RETRY_DELAY_SECS + random.uniform(0, 0.5))

        except RateLimitError as e:
            if attempt >= MAX_RETRIES:
                raise Exception(
                    f"Max retries ({MAX_RETRIES}) exceeded due to rate limiting"
                ) from e
            await asyncio.sleep(RETRY_DELAY_SECS + random.uniform(0.5, 1.5))

        except APIStatusError as e:
            # Retry 5xx; bail on other 4xx
            if 500 <= e.status_code < 600 and attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SECS + random.uniform(0, 0.5))
            else:
                raise Exception(
                    f"API error (status {e.status_code}) after {attempt} attempts"
                ) from e

        except Exception as e:
            # Re-raise unexpected exceptions
            raise Exception(f"Unexpected error after {attempt} attempts: {e!r}") from e
