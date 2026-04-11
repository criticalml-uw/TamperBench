"""Shared utility functions for evaluations.

This module provides reusable functions for common evaluation patterns:
- `format_chat_prompt`: Format messages for chat models with fallback
- `generate`: High-throughput generation with vLLM (sends all prompts at once)
- `generate_single`: Sequential generation for cases requiring dependencies between calls
- `llm_judge_score`: Score responses using an external LLM judge
- `strong_reject_score`: Score responses using the StrongReject classifier
- `aggregate_by_field`: Compute per-domain and overall metrics
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingTypeStubs=false

import logging
import re
from collections.abc import Callable
from typing import Any, cast

import openai
import polars as pl
from openai import OpenAI
from pandera.typing.polars import DataFrame
from strong_reject.evaluate import strongreject_finetuned as strong_reject_finetuned
from tqdm.contrib.concurrent import thread_map
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams

from tamperbench.whitebox.evals.output_schema import InferenceSchema, ScoreSchema

logger = logging.getLogger(__name__)


def format_chat_prompt(
    user_content: str,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Format messages for chat models with automatic fallback.

    Tries to use the tokenizer's chat_template if available, otherwise
    falls back to a simple text format.

    Args:
        user_content: The user's message content.
        tokenizer: The tokenizer to use for formatting.
        system_prompt: Optional system prompt to prepend.
        history: Optional conversation history (list of {"role": ..., "content": ...}).

    Returns:
        Formatted prompt string ready for model input.
    """
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_content})

    # Try chat template first
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return cast(str, tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    # Fallback for models without chat template
    parts = []
    for msg in messages:
        role = msg["role"].title()
        parts.append(f"{role}: {msg['content']}")

    return "\n\n".join(parts) + "\n\nAssistant:"


def llm_judge_score(
    inferences: DataFrame[InferenceSchema],
    prompt_template: str,
    score_pattern: str,
    model: str = "gpt-4o",
    max_workers: int = 16,
    default_score: float = 3.0,
    temperature: float = 0.0,
    max_tokens: int = 512,
    extra_fields: list[str] | None = None,
) -> DataFrame[ScoreSchema]:
    r"""Score responses using an external LLM judge.

    Args:
        inferences: DataFrame with 'prompt' and 'response' columns.
        prompt_template: Template string with placeholders for row fields.
            Must contain at least {prompt} and {response}.
            Can include additional fields if present in inferences.
        score_pattern: Regex pattern with one capture group for the score.
            Example: r"#thescore:\s*(\d+)" or r"Rating:\s*\[\[(\d+)\]\]"
        model: OpenAI model to use for judging.
        max_workers: Number of parallel API calls.
        default_score: Score to use if extraction fails.
        temperature: Temperature for judge model.
        max_tokens: Max tokens for judge response.
        extra_fields: Additional fields from inferences to preserve in output.

    Returns:
        DataFrame with original columns plus 'score' column.
    """
    client = OpenAI()
    extra_fields = extra_fields or []

    def score_one(row: dict[str, Any]) -> dict[str, Any]:
        judge_prompt = prompt_template.format(**row)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            match = re.search(score_pattern, content, re.IGNORECASE)
            if match:
                score = float(match.group(1))
            else:
                logger.warning(
                    "LLM judge score extraction failed (no pattern match), using default %.1f. Response: %s",
                    default_score,
                    content[:200],
                )
                score = default_score
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.APITimeoutError) as e:
            logger.warning(
                "LLM judge API call failed, using default %.1f. Error: %s",
                default_score,
                str(e),
            )
            score = default_score

        result: dict[str, Any] = {
            InferenceSchema.prompt: row[InferenceSchema.prompt],
            InferenceSchema.response: row[InferenceSchema.response],
            str(ScoreSchema.score): score,
        }
        # Preserve extra fields
        for field in extra_fields:
            if field in row:
                result[field] = row[field]

        return result

    rows = list(inferences.iter_rows(named=True))
    scores = thread_map(score_one, rows, max_workers=max_workers, desc="Scoring with LLM judge")

    return ScoreSchema.validate(pl.DataFrame(scores))


def strong_reject_score(
    inferences: DataFrame[InferenceSchema],
    batch_size: int = 32,
    max_response_length: int = 512,
    prompt_field: str = "prompt",
    extra_fields: list[str] | None = None,
) -> DataFrame[ScoreSchema]:
    """Score responses using the StrongReject finetuned classifier.

    Args:
        inferences: DataFrame with 'prompt' and 'response' columns.
        batch_size: Batch size for classifier.
        max_response_length: Max response length for classifier.
        prompt_field: Column name to use as the forbidden_prompt.
        extra_fields: Additional fields from inferences to preserve in output.

    Returns:
        DataFrame with original columns plus 'score' column.
    """
    extra_fields = extra_fields or []
    inferences_validated = InferenceSchema.validate(inferences)

    all_scores: list[float] = []
    n = len(inferences_validated)

    for i in range(0, n, batch_size):
        batch_prompts = list(inferences_validated[prompt_field][i : i + batch_size])
        batch_responses = list(inferences_validated[InferenceSchema.response][i : i + batch_size])

        batch_dict = {
            "forbidden_prompt": batch_prompts,
            "response": batch_responses,
        }

        batch_scores = strong_reject_finetuned(
            batch=batch_dict,
            max_response_length=max_response_length,
        )["score"]

        all_scores.extend(batch_scores)

    # Build output dict
    scores_dict: dict[str, Any] = inferences_validated.to_dict()
    scores_dict["score"] = pl.Series(all_scores)

    return ScoreSchema.validate(pl.from_dict(scores_dict))


def generate(
    prompts: list[str],
    model: LLM,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[str]:
    """Generate responses for all prompts at once using vLLM.

    This is the primary generation function - uses vLLM's continuous batching
    for high throughput. All prompts are submitted together and vLLM handles
    the batching internally.

    Args:
        prompts: List of prompt strings.
        model: vLLM LLM instance.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).

    Returns:
        List of generated response strings in the same order as prompts.
    """
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    outputs = model.generate(prompts, sampling_params)

    # vLLM returns outputs in the same order as inputs
    return [output.outputs[0].text.strip() for output in outputs]


def generate_single(
    prompt: str,
    model: LLM,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate a response for a single prompt.

    Use this only when prompts depend on previous outputs (e.g., multi-turn).
    For independent prompts, prefer `generate` which is more efficient.

    Args:
        prompt: The prompt string.
        model: vLLM LLM instance.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).

    Returns:
        Generated response string.
    """
    return generate([prompt], model, max_new_tokens, temperature)[0]


def aggregate_by_field(
    scores: DataFrame[ScoreSchema],
    field: str,
    score_field: str = "score",
) -> dict[str, float]:
    """Compute mean score per unique value in a field, plus overall.

    Args:
        scores: DataFrame with scores.
        field: Column name to group by (e.g., "domain", "category").
        score_field: Column name containing the scores.

    Returns:
        Dict mapping field values to mean scores, plus "overall" key.
    """
    metrics: dict[str, float] = {}

    for value in scores[field].unique().to_list():
        subset = scores.filter(pl.col(field) == value)
        if len(subset) > 0:
            metrics[str(value)] = cast(float, subset[score_field].mean())

    metrics["overall"] = cast(float, scores[score_field].mean())

    return metrics


def generate_with_logprobs(
    prompts: list[str],
    model: LLM,
    num_logprobs: int = 20,
) -> list[dict[str, Any]]:
    """Generate one token per prompt and return logprobs.

    Useful for multiple-choice evaluation where we need to compare
    probabilities of different answer tokens.

    Note: vLLM v1 limits ``logprobs`` to a maximum of 20. If an answer token
    is ranked beyond the top-20, it will not appear in logprobs and the
    fallback to generated text will be used instead. For standard A/B/C/D
    multiple-choice this is not an issue.

    Args:
        prompts: List of prompt strings.
        model: vLLM LLM instance.
        num_logprobs: Number of top logprobs to return per token (max 20 in vLLM v1).

    Returns:
        List of dicts, each with 'text' and 'logprobs' keys.
        logprobs is a dict mapping token strings to log probabilities.
    """
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=num_logprobs,
    )

    outputs = model.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        gen_output = output.outputs[0]
        text = gen_output.text.strip()

        # Extract logprobs dict from vLLM output
        logprobs_dict: dict[str, float] = {}
        if gen_output.logprobs:
            # vLLM returns list of logprobs dicts, one per token
            # We only generated 1 token, so take the first
            token_logprobs = gen_output.logprobs[0]
            # token_logprobs maps token_id to Logprob objects
            for logprob_obj in token_logprobs.values():
                # Logprob has .decoded_token and .logprob attributes
                token_str = logprob_obj.decoded_token
                if token_str is not None:
                    logprobs_dict[token_str] = logprob_obj.logprob

        results.append({"text": text, "logprobs": logprobs_dict})

    return results


def get_multiple_choice_answers(
    prompts: list[str],
    model: LLM,
    choices: list[str] | None = None,
) -> list[str]:
    """Get multiple-choice answers by comparing logprobs.

    For each prompt, finds which choice has the highest log probability.
    Falls back to the generated text if no choice is found in logprobs.

    Args:
        prompts: List of prompt strings (should end with "Answer:" or similar).
        model: vLLM LLM instance.
        choices: List of answer choices to compare.

    Returns:
        List of predicted answer strings (one per prompt).
    """
    # vLLM V1 limits logprobs to 20 max
    if choices is None:
        choices = ["A", "B", "C", "D"]
    outputs = generate_with_logprobs(prompts, model, num_logprobs=20)

    # Map lowercase variants to canonical choice for case-insensitive fallback
    choices_lower = {c.lower(): c for c in choices}

    answers = []
    for output in outputs:
        logprobs = output["logprobs"]
        generated_text = output["text"]

        # Find the choice with highest logprob
        best_choice = None
        best_logprob = float("-inf")

        for choice in choices:
            # Try multiple tokenization variants
            variants = [choice, f" {choice}", choice.lower(), f" {choice.lower()}"]
            for variant in variants:
                if variant in logprobs and logprobs[variant] > best_logprob:
                    best_logprob = logprobs[variant]
                    best_choice = choice

        # If no choice found in logprobs, fall back to generated text
        if best_choice is None:
            first_char = generated_text.strip()[:1].lower()
            if first_char in choices_lower:
                best_choice = choices_lower[first_char]
            else:
                # Last resort: default to first choice
                logger.warning(
                    "Multiple choice answer extraction failed, defaulting to '%s'. Generated text: %s",
                    choices[0],
                    generated_text[:100],
                )
                best_choice = choices[0]

        answers.append(best_choice)

    return answers


def create_score_extractor(
    patterns: list[str],
    default_score: float = 3.0,
    clamp: tuple[float, float] | None = None,
) -> Callable[[str], float]:
    r"""Create a function to extract scores from LLM judge responses.

    Args:
        patterns: List of regex patterns to try (each should have one capture group).
        default_score: Score to return if no pattern matches.
        clamp: Optional (min, max) tuple to clamp extracted scores.

    Returns:
        Function that takes a string and returns a float score.
    """

    def extract(text: str) -> float:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                if clamp:
                    score = max(clamp[0], min(clamp[1], score))
                return score
        return default_score

    return extract
