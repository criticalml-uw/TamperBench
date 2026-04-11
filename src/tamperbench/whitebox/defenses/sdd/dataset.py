"""Dataset construction for SDD defense.

Constructs paired data of <harmful prompt, high-quality irrelevant response>
by combining harmful prompts with benign responses, filtered for semantic
irrelevance using SentenceBERT embeddings.

The paper uses BeaverTails for harmful instructions (with category-balanced
sampling across 14 harm categories) and both Alpaca + LIMA for benign
responses. LIMA (GAIR/lima) is gated and requires accepting the license
on HuggingFace.

Reference implementation: https://github.com/ZeroNLP/SDD
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false

import random

import datasets
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _extract_harmful_prompts(
    dataset_name: str,
    split: str,
    num_samples: int,
    seed: int = 42,
) -> list[str]:
    """Extract harmful prompts with optional category-balanced sampling.

    Supports:
    - PKU-Alignment/BeaverTails: filters by ``is_safe=False``, deduplicates,
      and performs category-balanced sampling across 14 harm categories
      (matching the paper's methodology).
    - AdvBench variants (walledai/AdvBench, AlignmentResearch/AdvBench):
      uses random sampling.

    Args:
        dataset_name: HuggingFace dataset name.
        split: Dataset split to use.
        num_samples: Number of samples to extract.
        seed: Random seed for reproducibility.

    Returns:
        List of harmful prompt strings.
    """
    rng = random.Random(seed)
    dataset = datasets.load_dataset(dataset_name, split=split)
    if not isinstance(dataset, datasets.Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)}")

    if "BeaverTails" in dataset_name:
        return _extract_from_beavertails(dataset, num_samples, rng)

    # AdvBench format
    if "prompt" in dataset.column_names:
        all_prompts: list[str] = [sample["prompt"] for sample in dataset]
    elif "instructions" in dataset.column_names:
        all_prompts = [sample["instructions"] for sample in dataset]
    else:
        raise ValueError(
            f"Dataset {dataset_name} has no recognized prompt column. Available columns: {dataset.column_names}"
        )

    if len(all_prompts) == 0:
        raise ValueError(f"No prompts found in {dataset_name}/{split}")

    if num_samples <= len(all_prompts):
        return rng.sample(all_prompts, num_samples)
    else:
        return rng.choices(all_prompts, k=num_samples)


def _extract_from_beavertails(
    dataset: datasets.Dataset,
    num_samples: int,
    rng: random.Random,
) -> list[str]:
    """Extract harmful prompts from BeaverTails with category-balanced sampling.

    Matches the paper's methodology: "balanced by sampling equal amounts from
    each category" across BeaverTails' 14 harm categories.

    Args:
        dataset: Loaded BeaverTails dataset.
        num_samples: Target number of prompts.
        rng: Seeded random instance.

    Returns:
        List of harmful prompt strings, balanced across categories.
    """
    # Collect unsafe prompts grouped by primary category
    prompts_by_category: dict[str, list[str]] = {}

    for row in dataset:
        if row["is_safe"]:
            continue
        prompt = row["prompt"]

        # category is a dict mapping category names to booleans (multi-label)
        category_dict = row["category"]
        if isinstance(category_dict, dict):
            primary_cat = next(
                (cat for cat, val in category_dict.items() if val),
                "unknown",
            )
        else:
            primary_cat = str(category_dict) if category_dict else "unknown"

        prompts_by_category.setdefault(primary_cat, []).append(prompt)

    if not prompts_by_category:
        raise ValueError("No unsafe prompts found in BeaverTails dataset")

    # Category-balanced sampling: equal amounts from each category
    categories = sorted(prompts_by_category.keys())
    per_category = num_samples // len(categories)

    selected: list[str] = []
    overflow: list[str] = []

    for cat in categories:
        pool = prompts_by_category[cat][:]
        rng.shuffle(pool)
        take = min(per_category, len(pool))
        selected.extend(pool[:take])
        overflow.extend(pool[take:])

    # Fill remaining slots (there is a remainder from the integer division, and
    # if a category is smaller than per_category in length there will also be
    # extra slots).
    shortfall = num_samples - len(selected)
    if shortfall > 0:
        rng.shuffle(overflow)
        assert len(overflow) >= shortfall, (
            f"Not enough prompts: need {num_samples}, but only {len(selected) + len(overflow)} available"
        )
        selected.extend(overflow[:shortfall])

    rng.shuffle(selected)
    return selected


def _extract_benign_responses(
    dataset_names: list[str],
    split: str,
) -> list[str]:
    """Extract high-quality responses from one or more benign datasets.

    Supports Alpaca ('output' column), LIMA ('conversations' column),
    and any dataset with a 'response' column. The paper uses both
    Alpaca and LIMA.

    Args:
        dataset_names: List of HuggingFace dataset names.
        split: Dataset split to use (applied to all datasets).

    Returns:
        List of high-quality response strings pooled from all datasets.
    """
    responses: list[str] = []
    for dataset_name in dataset_names:
        print(f"      Loading {dataset_name}...")
        dataset = datasets.load_dataset(dataset_name, split=split)
        if not isinstance(dataset, datasets.Dataset):
            raise TypeError(f"Expected Dataset, got {type(dataset)}")

        count_before = len(responses)
        for sample in dataset:
            # LIMA uses 'conversations' format: list of [user, assistant, ...]
            if "conversations" in sample:
                convs = sample["conversations"]
                # Extract assistant responses (odd indices)
                for i in range(1, len(convs), 2):
                    responses.append(convs[i])
            # Alpaca uses 'output' field
            elif "output" in sample:
                if sample["output"].strip():
                    responses.append(sample["output"])
            # Fallback: try 'response' field
            elif "response" in sample:
                if sample["response"].strip():
                    responses.append(sample["response"])

        print(f"      -> {len(responses) - count_before} responses from {dataset_name}")

    return responses


def construct_sdd_dataset(
    harmful_dataset_name: str,
    harmful_dataset_split: str,
    benign_dataset_names: list[str],
    benign_dataset_split: str,
    num_samples: int,
    similarity_threshold: float,
    sentence_transformer_model: str,
    max_resample_attempts: int = 50,
    seed: int = 42,
) -> datasets.Dataset:
    """Construct SDD training dataset with irrelevance filtering.

    Creates pairs of <harmful prompt, high-quality irrelevant response> by:
    1. Loading harmful prompts (from BeaverTails, or AdvBench)
    2. Loading high-quality responses from Alpaca and LIMA
    3. Random pairing with cosine similarity filtering
    4. Fallback to best-seen response if threshold not met

    Reference: https://github.com/ZeroNLP/SDD

    Args:
        harmful_dataset_name: HuggingFace dataset for harmful prompts.
        harmful_dataset_split: Split to use for harmful prompts.
        benign_dataset_names: HuggingFace datasets for benign responses.
        benign_dataset_split: Split to use for all benign response datasets.
        num_samples: Number of paired samples to construct.
        similarity_threshold: Maximum cosine similarity allowed (default: 0.25).
            Pairs above this threshold are resampled to ensure irrelevance.
        sentence_transformer_model: Model name for SentenceBERT.
        max_resample_attempts: Max attempts to find irrelevant response (default: 50).
        seed: Random seed for reproducibility.

    Returns:
        datasets.Dataset: Dataset with 'prompt', 'response', and 'cosine_sim' columns.
    """
    rng = random.Random(seed)

    print(f"   Loading harmful prompts from {harmful_dataset_name}...")
    harmful_prompts = _extract_harmful_prompts(
        harmful_dataset_name,
        harmful_dataset_split,
        num_samples,
        seed=seed,
    )
    print(f"   Loaded {len(harmful_prompts)} harmful prompts")

    print(f"   Loading benign responses from {benign_dataset_names}...")
    benign_responses = _extract_benign_responses(
        benign_dataset_names,
        benign_dataset_split,
    )
    print(f"   Loaded {len(benign_responses)} total benign responses")

    if len(benign_responses) < 100:
        raise ValueError(
            f"Insufficient benign responses ({len(benign_responses)}). Need at least 100 for effective resampling."
        )

    print(f"   Loading SentenceBERT model ({sentence_transformer_model})...")
    model = SentenceTransformer(sentence_transformer_model)

    # Precompute all benign response embeddings in batch. Normalized embeddings
    # allow cosine similarity via dot product.
    print("   Precomputing benign response embeddings...")
    benign_embeddings = model.encode(
        benign_responses,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"   Constructing {num_samples} pairs with irrelevance filtering...")
    paired_data: list[dict[str, str | float | int]] = []
    total_resamples = 0
    threshold_failures = 0

    for prompt in tqdm(harmful_prompts, desc="   Pairing"):
        # Encode the harmful prompt once (normalized for dot-product similarity)
        prompt_embedding = model.encode(
            [prompt],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Track best (lowest similarity) response seen during resampling
        best_idx: int | None = None
        best_similarity: float = float("inf")
        resample_count = 0

        for _ in range(max_resample_attempts):
            idx = rng.randrange(len(benign_responses))
            similarity = float(np.dot(prompt_embedding, benign_embeddings[idx]))

            # Track best seen
            if similarity < best_similarity:
                best_similarity = similarity
                best_idx = idx

            # Found one below threshold - use it
            if similarity <= similarity_threshold:
                break

            resample_count += 1

        # Use best seen response (fallback if threshold not met)
        assert best_idx is not None
        final_response = benign_responses[best_idx]
        final_similarity = best_similarity

        if best_similarity > similarity_threshold:
            threshold_failures += 1

        total_resamples += resample_count
        paired_data.append(
            {
                "prompt": prompt,
                "response": final_response,
                "cosine_sim": final_similarity,
                "resample_tries": resample_count,
            }
        )

    print(f"   Total resamples: {total_resamples}")
    print(f"   Threshold failures (used fallback): {threshold_failures}")
    print(f"   Final dataset size: {len(paired_data)}")

    return datasets.Dataset.from_list(paired_data)
