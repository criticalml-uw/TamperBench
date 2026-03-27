"""Clean data curation for CTRL defense with simple pipeline and automatic caching."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datasets
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import LLM, SamplingParams

from tamperbench.whitebox.defenses.ctrl.data_types import Input, Prompt, Variant
from tamperbench.whitebox.defenses.ctrl.prompts_cache import Cache
from tamperbench.whitebox.defenses.ctrl.worker import parallelize_across_gpus
from tamperbench.whitebox.utils.datasets import (
    get_config_cache_path,
    get_dataset,
    save_cached_dataset,
)
from tamperbench.whitebox.utils.metrics import (
    compute_helpfulness,
    compute_perplexity,
    compute_readability,
    ensure_readability_nltk_resources,
)

if TYPE_CHECKING:
    from tamperbench.whitebox.defenses.ctrl.ctrl import CTRLConfig

logger = logging.getLogger(__name__)

# System prompt for curation.
# The paper (Section 4.2) specifies the prompt: "Given a query and response, revise the
# response statements to present an alternative perspective in answering the query."
# We deliberately diverge from this: empirical testing of ~30 prompt variations found that
# a simpler "clearer and more helpful" framing produces better rephrasings (more coherent,
# less meta-commentary) than the paper's "alternative perspective" wording.
CURATION_SYSTEM_PROMPT = (
    "You rewrite responses to be clearer and more helpful. "
    "Output ONLY the rewritten response text, nothing else. "
    "Do not include phrases like 'Here is' or 'Improved response'."
)

# User prompt template for curation - designed to elicit direct response, not meta-commentary
CURATION_USER_TEMPLATE = """Rewrite this response to be clearer and more helpful.

Question: {query}

Response to rewrite: {response}

Rewritten response:"""


def format_curation_prompt(query: str, response: str, tokenizer: PreTrainedTokenizer) -> str:
    """Format curation prompt with the model's native chat template.

    Uses tokenizer.apply_chat_template() to produce correctly-formatted prompts
    for any model (Llama 2, Llama 3, Qwen, etc.) without hardcoded templates.

    Args:
        query: The original query.
        response: The response to improve.
        tokenizer: Tokenizer for the model (used for chat template).

    Returns:
        Formatted prompt string with chat template applied.
    """
    user_content = CURATION_USER_TEMPLATE.format(query=query, response=response)

    messages = [
        {"role": "system", "content": CURATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    result = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(result, str)
    return result


# Stop sequences - avoid truncating too early while preventing runaway generation
# Be careful not to include sequences that appear in normal content (like "---" in narratives)
CURATION_STOP_SEQUENCES = [
    "\n\nQuestion:",  # Start of next example
    "\n\nRewrite",  # Start of next instruction
    "\nQuestion:",
    "\n---\n",  # Only catch dividers on their own line, not inline dashes
    "\n## ",  # Markdown headers at line start only
    "</s>",  # EOS token (text form)
    "<|eot_id|>",  # Llama 3 end of turn
]

# Prefixes to strip from generated responses (meta-commentary that slipped through)
# These are checked as prefixes with case-insensitive matching
META_PREFIXES = [
    "Improved response:",
    "Revised response:",
    "Rewritten response:",
    "Sure, here",
    "Sure! Here",
    "Certainly! Here",
    "Certainly, here",
    "Of course! Here",
    "Of course, here",
]

# Patterns that indicate meta-commentary - if line starts with these, skip that line
META_LINE_STARTS = [
    "here's a",
    "here is a",
    "here's the",
    "here is the",
    "here's an",
    "here is an",
    "i've rewritten",
    "i have rewritten",
    "below is",
]


def clean_generated_response(text: str) -> str:
    """Clean up generated response by removing meta-commentary prefixes.

    Args:
        text: Raw generated text.

    Returns:
        Cleaned text with meta-prefixes removed.
    """
    text = text.strip()
    if not text:
        return text

    # Remove common meta-prefixes (case-insensitive check, preserve case in result)
    text_lower = text.lower()
    for prefix in META_PREFIXES:
        if text_lower.startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
            text_lower = text.lower()

    # Check if first line is meta-commentary and skip it
    lines = text.split("\n", 1)
    first_line_lower = lines[0].lower().strip()
    for pattern in META_LINE_STARTS:
        if first_line_lower.startswith(pattern):
            # Skip the first line if there's more content on next line
            if len(lines) > 1 and lines[1].strip():
                text = lines[1].strip()
            # Or if there's a colon, take content after it (same line)
            elif ":" in lines[0]:
                after_colon = lines[0].split(":", 1)[1].strip()
                if after_colon:
                    text = after_colon
                    if len(lines) > 1:
                        text = text + "\n" + lines[1]
            break

    # Remove leading quotes if present
    if text.startswith('"') and text.count('"') >= 2:
        # Find closing quote and extract content
        end_quote = text.rfind('"')
        if end_quote > 0:
            text = text[1:end_quote]

    return text.strip()


def generate_all_variants(
    inputs: list[Input],
    vllm_model: LLM,
    temperatures: Sequence[float],
    top_ps: Sequence[float],
    cache: Cache,
    tokenizer: PreTrainedTokenizer,
) -> list[Variant]:
    """Generate all possible variants for all inputs using all temp/top_p combinations.

    Checks cache first and only generates variants that don't already exist.
    GPU operation. Batches everything together.

    Args:
        inputs: List of Input objects to generate variants for
        vllm_model: vLLM model
        temperatures: Temperature values to try
        top_ps: Top-p values to try
        cache: Cache to check for existing variants
        tokenizer: Tokenizer for the model (used for chat template formatting)

    Returns:
        List of Variant objects (without metrics computed)
    """
    if not inputs:
        return []

    # Build list of (prompt, source_hash) tuples
    queries_with_source: list[tuple[Prompt, str]] = []
    for inp in inputs:
        source_hash = Variant.hash_source(inp.response)
        for temp in temperatures:
            for top_p in top_ps:
                prompt = Prompt(
                    query=inp.query,
                    prompt=format_curation_prompt(inp.query, inp.response, tokenizer),
                    sampling_params=SamplingParams(
                        temperature=temp,
                        top_p=top_p,
                        max_tokens=512,  # Paper says 256, but that truncates ~7% of longer responses
                        stop=CURATION_STOP_SEQUENCES,
                    ),
                )
                queries_with_source.append((prompt, source_hash))

    variants = []
    missing_variants: list[tuple[Prompt, str]] = []
    for prompt, source_hash in queries_with_source:
        if var := cache.get_variant(
            prompt.query,
            prompt.sampling_params.temperature,
            prompt.sampling_params.top_p,
            source_hash,
        ):
            variants.append(var)
        else:
            missing_variants.append((prompt, source_hash))

    # Generate everything at once
    outputs = vllm_model.generate(
        [prompt.prompt for prompt, _ in missing_variants],
        [prompt.sampling_params for prompt, _ in missing_variants],
    )

    new_variants = []
    for output, (prompt, source_hash) in zip(outputs, missing_variants, strict=True):
        text = clean_generated_response(output.outputs[0].text)
        if text:
            new_variants.append(
                Variant(
                    query=prompt.query,
                    text=text,
                    temp=prompt.sampling_params.temperature,
                    top_p=prompt.sampling_params.top_p,
                    source_hash=source_hash,
                )
            )

    return variants + new_variants


def compute_metrics(
    queries: list[str],
    texts: list[str],
    vllm_model: LLM,
) -> list[tuple[float, float, float | None]]:
    """Compute all metrics for query-response pairs.

    Pure function that batches GPU operations. Returns metrics in same order as inputs.

    Args:
        queries: List of queries
        texts: List of response texts
        vllm_model: vLLM model

    Returns:
        List of (perplexity, readability, helpfulness | None) tuples
    """
    tqdm.write(f"  Computing perplexity for {len(queries)} queries...")
    perplexities = compute_perplexity(texts, vllm_model)
    tqdm.write(f"  Computing readability for {len(queries)} queries...")
    readabilities = compute_readability(texts)
    tqdm.write(f"  Computing helpfulness for {len(queries)} queries...")
    helpfulnesses = compute_helpfulness(queries, texts, vllm_model)

    return [
        (ppl, read, help_score)
        for ppl, read, help_score in zip(perplexities, readabilities, helpfulnesses, strict=True)
    ]


def add_metrics(variants: list[Variant], vllm_model: LLM) -> list[Variant]:
    """Compute metrics for variants that don't have them yet.

    Pure function that returns new variants with metrics filled.

    Args:
        variants: Variants (may or may not have metrics)
        vllm_model: vLLM model

    Returns:
        Variants with all metrics computed
    """
    with_metrics = [v for v in variants if v.has_metrics()]
    without_metrics = [v for v in variants if not v.has_metrics()]

    # Compute metrics for those that need it
    if not without_metrics:
        return with_metrics

    tqdm.write(f"  Computing metrics for {len(without_metrics)} variants...")
    queries = [v.query for v in without_metrics]
    texts = [v.text for v in without_metrics]

    metrics = compute_metrics(queries, texts, vllm_model)
    scored = []
    dropped_count = 0
    for v, (ppl, read, help_score) in zip(without_metrics, metrics, strict=True):
        if help_score is None:
            dropped_count += 1
            continue
        scored.append(
            Variant(
                query=v.query,
                text=v.text,
                temp=v.temp,
                top_p=v.top_p,
                perplexity=ppl,
                readability=read,
                helpfulness=help_score,
                source_hash=v.source_hash,
            )
        )
    if dropped_count:
        logger.warning(
            "Dropped %d/%d variants due to failed helpfulness scoring",
            dropped_count,
            len(without_metrics),
        )
    return with_metrics + scored


def filter_variants(
    variants: list[Variant],
    original_metrics: dict[str, tuple[float, float]],
    min_readability_ratio: float = 0.1,
    min_helpfulness_ratio: float = 0.1,
) -> list[Variant]:
    """Filter variants by quality thresholds.

    Note: The specific threshold values (0.1) are implementation choices as the paper
    only mentions filtering by "text quality" without specifying exact thresholds.

    Args:
        variants: Variants to filter
        original_metrics: Dict of query -> (readability, helpfulness)
        min_readability_ratio: Minimum readability as fraction of original (default: 0.1)
        min_helpfulness_ratio: Minimum helpfulness as fraction of original (default: 0.1)

    Returns:
        Filtered list of variants
    """
    result = []
    for v in variants:
        if not v.has_metrics():
            continue

        # Look up original metrics by query (which stays constant across rounds)
        orig_read, orig_help = original_metrics.get(v.query, (0.0, 0.0))
        if orig_read == 0 or orig_help == 0:
            continue

        # Type narrowing: has_metrics() guarantees these are not None
        assert v.readability is not None
        assert v.helpfulness is not None

        if v.readability >= orig_read * min_readability_ratio and v.helpfulness >= orig_help * min_helpfulness_ratio:
            result.append(v)

    return result


def select_best_per_input(
    variants: list[Variant],
    beam_width: int,
) -> dict[str, list[Variant]]:
    """Select top-k variants per input by perplexity.

    Args:
        variants: Variants to select from
        beam_width: Number to keep per input

    Returns:
        Dict of query -> list of best variants
    """
    # Group by query
    by_query: dict[str, list[Variant]] = {}
    for v in variants:
        if v.query not in by_query:
            by_query[v.query] = []
        by_query[v.query].append(v)

    # Sort each group by perplexity and take top-k
    result = {}
    for query, variants_list in by_query.items():
        sorted_variants = sorted(variants_list, key=lambda x: x.perplexity or float("inf"))
        result[query] = sorted_variants[:beam_width]

    return result


def inputs_to_variants(
    inputs: list[Input],
    vllm_model: LLM,
    cache: Cache,
) -> list[Variant]:
    """Convert inputs to variants with metrics computed.

    Args:
        inputs: Input objects to convert
        vllm_model: vLLM model for computing metrics
        cache: Cache instance

    Returns:
        Variants with all metrics computed
    """
    # Check cache first for existing variants with metrics
    variants = []
    for inp in inputs:
        # Initial variants have temp=0, top_p=0, source_hash=None
        cached = cache.get_variant(inp.query, 0.0, 0.0, None)
        if cached and cached.has_metrics():
            variants.append(cached)
        else:
            variants.append(
                Variant(
                    query=inp.query,
                    text=inp.response,
                    temp=0.0,
                    top_p=0.0,
                )
            )

    variants = add_metrics(variants, vllm_model)
    cache.save_variants(variants)
    return variants


def get_best_variants(
    current_variants: list[Variant],
    vllm_model: LLM,
    cache: Cache,
    temperatures: Sequence[float],
    top_ps: Sequence[float],
    beam_width: int,
    original_metrics: dict[str, tuple[float, float]],
    tokenizer: PreTrainedTokenizer,
) -> list[Variant]:
    """One round of beam search: generate new variants, filter, select best.

    Pure function: variants -> better variants.

    Args:
        current_variants: Current best variants (with metrics)
        vllm_model: vLLM model
        cache: Cache instance
        temperatures: Temperature values for generation
        top_ps: Top-p values for generation
        beam_width: Number of variants to keep per query
        original_metrics: Original (readability, helpfulness) per query for filtering
        tokenizer: Tokenizer for the model (used for chat template formatting)

    Returns:
        Best variants after this round (with metrics)
    """
    # Convert current variants back to Input format for generation
    current_as_inputs = [Input(query=v.query, response=v.text) for v in current_variants]

    # Generate new variants from current
    tqdm.write(f"  Generating variants for {len(current_as_inputs)} inputs...")
    new_variants = generate_all_variants(current_as_inputs, vllm_model, temperatures, top_ps, cache, tokenizer)
    cache.save_variants(new_variants)

    # Combine current + new
    all_variants = current_variants + new_variants

    # Compute metrics for any missing
    all_variants = add_metrics(all_variants, vllm_model)
    cache.save_variants(all_variants)

    # Filter by quality thresholds (from paper)
    tqdm.write(f"  Filtering {len(all_variants)} variants...")
    filtered = filter_variants(all_variants, original_metrics)
    cache.save_variants(filtered)
    tqdm.write(f"  {len(filtered)} variants passed filters")

    # Select best by perplexity
    best_per_query = select_best_per_input(filtered, beam_width)

    # Flatten to list
    result = []
    for variants_list in best_per_query.values():
        result.extend(variants_list)

    return result


def curate_batch(
    inputs: list[Input],
    model_path: str,
    cache_dir: Path,
    num_rounds: int,
    temperatures: Sequence[float],
    top_ps: Sequence[float],
    beam_width: int,
) -> list[tuple[str, str]]:
    """Curate a batch of inputs through multiple rounds of beam search.

    Loads its own model and cache, suitable for parallel execution.

    Args:
        inputs: List of inputs to curate
        model_path: Path to model checkpoint
        cache_dir: Directory for cache storage
        num_rounds: Number of refinement rounds
        temperatures: Temperature values
        top_ps: Top-p values
        beam_width: Beam width for search

    Returns:
        List of (query, curated_response) tuples
    """
    cache = Cache(cache_dir)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path)
    # Base models (e.g. Meta-Llama-3-8B) lack a chat template; set a minimal
    # fallback so apply_chat_template works during curation prompt formatting.
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}### Instruction:\n{{ message['content'] }}\n\n"
            "{% elif message['role'] == 'assistant' %}### Response:\n{{ message['content'] }}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}### Response:\n{% endif %}"
        )
    vllm_model = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )

    # Convert initial inputs to variants with metrics (round 0)
    # Cache lookup happens inside - if already computed, this is fast
    tqdm.write("Round 0 (initial)")
    current_variants = inputs_to_variants(inputs, vllm_model, cache)
    cache.save_variants(current_variants)

    # Extract original metrics from initial variants for filtering
    original_metrics = {
        v.query: (v.readability, v.helpfulness)
        for v in current_variants
        if v.readability is not None and v.helpfulness is not None
    }

    # Run beam search rounds
    # Cache lookups happen inside each round - cached variants/metrics are reused
    for round_idx in range(num_rounds):
        tqdm.write(f"\nRound {round_idx + 1}/{num_rounds}")
        current_variants = get_best_variants(
            current_variants,
            vllm_model,
            cache,
            temperatures,
            top_ps,
            beam_width,
            original_metrics,
            tokenizer,
        )

    # Extract final results (one variant per query)
    # Group by query and take first (already sorted by perplexity)
    by_query: dict[str, Variant] = {}
    for v in current_variants:
        if v.query not in by_query:
            by_query[v.query] = v

    # Return in original order
    result = []
    for inp in inputs:
        if inp.query in by_query:
            result.append((inp.query, by_query[inp.query].text))
        else:
            # Fallback to original if something went wrong
            result.append((inp.query, inp.response))

    return result


def extract_query_response(sample: dict[str, Any]) -> tuple[str, str]:
    """Extract query and response from dataset sample."""
    if "instruction" in sample and "output" in sample:
        query = sample["instruction"]
        if sample.get("input"):
            query = f"{query}\n\n{sample['input']}"
        response = sample["output"]
    elif "prompt" in sample and "response" in sample:
        query = sample["prompt"]
        response = sample["response"]
    elif "question" in sample and "answer" in sample:
        query = sample["question"]
        response = sample["answer"]
    elif "text" in sample:
        text = sample["text"]
        if "### Response:" in text:
            parts = text.split("### Response:")
            query = parts[0].replace("### Instruction:", "").strip()
            response = parts[1].strip()
        else:
            raise ValueError(
                f"'text' field has no recognized structure (e.g. '### Response:'). "
                f"Cannot split into query/response. Text starts with: {text[:100]!r}"
            )
    else:
        raise ValueError(f"Unknown dataset format. Available keys: {list(sample.keys())}")

    return query, response


def curate_dataset(
    config: "CTRLConfig",
    raw_dataset: datasets.Dataset | None = None,
) -> datasets.Dataset:
    """Curate a dataset using CTRL with automatic caching and parallelization.

    Args:
        config: CTRL configuration
        raw_dataset: Pre-loaded raw dataset. If None, loads from config. Pass this
            to avoid loading the dataset a second time when it was already loaded
            by the caller.

    Returns:
        Curated dataset
    """
    # Build cache config (includes model so different models get separate caches)
    cache_config = {
        "model_name": str(config.input_checkpoint_path),
        "dataset_name": config.clean_dataset_name,
        "dataset_split": config.clean_dataset_split,
        "curation_rate": config.curation_rate,
        "num_samples": config.num_samples,
        "temperatures": config.temperatures,
        "top_ps": config.top_ps,
        "beam_search_rounds": config.beam_search_rounds,
        "beam_width": config.beam_width,
    }

    # Check if curated dataset is cached
    cache_key = f"ctrl_curated_{config.clean_dataset_name}"
    cache_path = get_config_cache_path(cache_key, cache_config)

    if cache_path.exists():
        print(f"   Loading cached curated dataset from: {cache_path}")
        return datasets.Dataset.from_list(get_dataset(cache_path, data_format="jsonl"))

    print("   Starting curation...")

    # Load dataset if not provided
    if raw_dataset is None:
        raw_dataset = datasets.load_dataset(  # type: ignore[assignment]
            config.clean_dataset_name,
            split=config.clean_dataset_split,
        )

    # Select subset
    if len(raw_dataset) > config.num_samples:
        raw_dataset = raw_dataset.select(range(config.num_samples))

    # Determine items to curate
    num_to_curate = int(len(raw_dataset) * config.curation_rate)
    print(f"   Curating {num_to_curate} out of {len(raw_dataset)} samples...")

    if num_to_curate == 0:
        logger.warning(
            "num_to_curate is 0 (dataset size=%d, curation_rate=%f). Returning empty dataset.",
            len(raw_dataset),
            config.curation_rate,
        )
        return datasets.Dataset.from_list([])

    indices_to_curate = list(range(0, len(raw_dataset), len(raw_dataset) // num_to_curate))[:num_to_curate]

    # Extract inputs
    inputs = [Input(*extract_query_response(raw_dataset[idx])) for idx in indices_to_curate]

    # Setup cache (sanitize key for filesystem)
    safe_cache_key = cache_key.replace("/", "_").replace(":", "_")
    cache_dir = cache_path.parent / f"{safe_cache_key}_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Cache directory: {cache_dir}")

    num_gpus = torch.cuda.device_count()
    print(f"   Found {num_gpus} GPU(s)")

    ensure_readability_nltk_resources()

    # Parallelize across available GPUs (or run directly if single GPU/CPU)
    curated_pairs = parallelize_across_gpus(
        items=inputs,
        function=curate_batch,
        function_kwargs={
            "model_path": str(config.input_checkpoint_path),
            "cache_dir": cache_dir,
            "num_rounds": config.beam_search_rounds,
            "temperatures": config.temperatures,
            "top_ps": config.top_ps,
            "beam_width": config.beam_width,
        },
        temp_dir=cache_dir / "parallel_temp",
    )

    # Build dataset
    curated_samples = [
        {
            "prompt": query,
            "response": response,
            "original_response": inputs[i].response,
        }
        for i, (query, response) in enumerate(curated_pairs)
    ]

    curated_dataset = datasets.Dataset.from_list(curated_samples)

    # Save to cache
    save_cached_dataset(curated_dataset, cache_key, cache_config)
    print(f"   Saved curated dataset to: {cache_path}")

    return curated_dataset
