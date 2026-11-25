"""Dataset construction for DeepAlign (safety-recovery data augmentation).

Every training example is a token sequence ``input_ids`` plus ``labels`` of the
same length. Positions labelled ``-100`` are ignored by the cross-entropy loss,
so the model is only trained to *produce* the tokens that keep their real id
in ``labels``.

Three kinds of examples are built:

1. Plain refusal (harmful prompt -> refusal)::

       input_ids: [prompt ............][refusal ........ EOS]
       labels:    [-100 .... -100     ][refusal ........ EOS]

2. Safety recovery (harmful prompt -> first k harmful tokens -> refusal)::

       input_ids: [prompt ......][harmful[:k] ...][refusal ........ EOS]
       labels:    [-100 ... -100][-100 ...... -100][refusal ........ EOS]

   The harmful prefix is *seen* as context but never trained on, so the model
   learns "even if my reply has already started going wrong, steer back to a
   refusal", which is what makes the alignment deeper than the first few tokens.

3. Benign utility (benign prompt -> helpful answer), same layout as (1).
   Mixed in so the model keeps its general capabilities and doesn't learn to
   refuse everything.
"""

import random
from typing import Any, cast

import datasets
from transformers import PreTrainedTokenizer

from tamperbench.whitebox.defenses.booster.data import clean_refusal_response

IGNORE_INDEX = -100

# Fallback prompt format for base models without a chat template (e.g. tiny test models).
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def token_ids(tokenizer: PreTrainedTokenizer, text: str, add_special_tokens: bool = False) -> list[int]:
    """Tokenize ``text`` into a plain list of token ids."""
    return cast(list[int], tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


def encode_prompt(tokenizer: PreTrainedTokenizer, user_message: str) -> list[int]:
    """Tokenize a user message up to the point where the assistant starts replying.

    With a chat template this is e.g. ``<s>[INST] question [/INST]``; the
    ``add_generation_prompt=True`` part appends the "assistant's turn" marker.

    Args:
        tokenizer: Tokenizer of the model being aligned.
        user_message: The user's request.

    Returns:
        Token ids of the prompt (including BOS if the model uses one).
    """
    if tokenizer.chat_template is not None:
        text = cast(
            str,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=True,
            ),
        )
        # The chat template already inserts BOS, so don't add it a second time.
        return token_ids(tokenizer, text)

    text = ALPACA_TEMPLATE.format(instruction=user_message)
    return token_ids(tokenizer, text, add_special_tokens=True)


def build_example(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    target: str,
    max_seq_length: int,
    harmful_prefix: str | None = None,
    num_prefix_tokens: int = 0,
) -> dict[str, list[int]] | None:
    """Build one training example in token space.

    Everything is tokenized separately and concatenated as token ids, so the
    boundary between "masked" and "trained" is exact. (Doing ``len()`` on the
    joined *string* would count characters, not tokens.)

    Args:
        tokenizer: Tokenizer of the model being aligned.
        prompt: User message.
        target: The text the model should learn to produce (refusal or helpful answer).
        max_seq_length: Sequences are truncated to this many tokens.
        harmful_prefix: Harmful response to take the first tokens from (recovery examples only).
        num_prefix_tokens: How many tokens of ``harmful_prefix`` to prepend (``k`` in the paper).

    Returns:
        Dict with ``input_ids``, ``labels`` and ``attention_mask``, or ``None``
        if truncation removed every trained token.
    """
    assert isinstance(tokenizer.eos_token_id, int)

    prompt_ids = encode_prompt(tokenizer, prompt)

    prefix_ids: list[int] = []
    if harmful_prefix and num_prefix_tokens > 0:
        prefix_ids = token_ids(tokenizer, harmful_prefix)[:num_prefix_tokens]

    target_ids = [*token_ids(tokenizer, target), tokenizer.eos_token_id]

    context_ids = prompt_ids + prefix_ids
    input_ids = (context_ids + target_ids)[:max_seq_length]
    labels = ([IGNORE_INDEX] * len(context_ids) + target_ids)[:max_seq_length]

    if all(label == IGNORE_INDEX for label in labels):
        return None

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def load_harmful_triples(dataset_name: str, split: str, num_samples: int, seed: int) -> list[dict[str, str]]:
    """Load (prompt, harmful response, refusal) triples.

    Expects the column layout of ``anonymous4486/repnoise_beavertail``:
    ``prompt``, ``response`` (harmful) and ``refusal`` (safe, sometimes in
    ``"Question: ... Answer: ..."`` form, which is cleaned up).

    Args:
        dataset_name: HuggingFace dataset name.
        split: Dataset split.
        num_samples: Number of triples to return.
        seed: Shuffle seed.

    Returns:
        List of dicts with keys ``prompt``, ``harmful`` and ``refusal``.
    """
    dataset = datasets.load_dataset(dataset_name, split=split)
    assert isinstance(dataset, datasets.Dataset)

    if "is_safe" in dataset.column_names:
        dataset = dataset.filter(lambda row: not row["is_safe"])  # pyright: ignore[reportUnknownLambdaType]

    dataset = dataset.shuffle(seed=seed)

    triples: list[dict[str, str]] = []
    for row in dataset:
        row = cast(dict[str, Any], row)
        refusal = clean_refusal_response(row.get("refusal") or "")
        harmful = row.get("response") or ""
        if row["prompt"].strip() and harmful.strip() and refusal.strip():
            triples.append({"prompt": row["prompt"], "harmful": harmful, "refusal": refusal})
        if len(triples) >= num_samples:
            break
    return triples


def load_benign_pairs(dataset_name: str, split: str, num_samples: int, seed: int) -> list[dict[str, str]]:
    """Load (prompt, helpful response) pairs from an Alpaca-style dataset.

    Args:
        dataset_name: HuggingFace dataset name with ``instruction``/``input``/``output`` columns.
        split: Dataset split.
        num_samples: Number of pairs to return.
        seed: Shuffle seed.

    Returns:
        List of dicts with keys ``prompt`` and ``response``.
    """
    if num_samples <= 0:
        return []

    dataset = datasets.load_dataset(dataset_name, split=split)
    assert isinstance(dataset, datasets.Dataset)
    dataset = dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))

    pairs: list[dict[str, str]] = []
    for row in dataset:
        row = cast(dict[str, Any], row)
        prompt = row["instruction"]
        if row.get("input"):
            prompt = f"{prompt}\n\n{row['input']}"
        pairs.append({"prompt": prompt, "response": row["output"]})
    return pairs


def build_deep_align_dataset(
    tokenizer: PreTrainedTokenizer,
    harmful_triples: list[dict[str, str]],
    benign_pairs: list[dict[str, str]],
    max_prefix_tokens: int,
    augment_prob: float,
    max_seq_length: int,
    seed: int = 42,
) -> datasets.Dataset:
    """Turn raw triples/pairs into a shuffled, tokenized training set.

    For each harmful triple, a coin flip with probability ``augment_prob``
    decides between a safety-recovery example (random ``k`` in
    ``[1, max_prefix_tokens]``) and a plain refusal example.

    Args:
        tokenizer: Tokenizer of the model being aligned.
        harmful_triples: Output of :func:`load_harmful_triples`.
        benign_pairs: Output of :func:`load_benign_pairs`.
        max_prefix_tokens: Upper bound on ``k``.
        augment_prob: Probability of making a recovery example instead of a plain refusal.
        max_seq_length: Truncation length.
        seed: RNG seed for the coin flips and ``k``.

    Returns:
        Dataset with ``input_ids``, ``labels`` and ``attention_mask`` columns.
    """
    rng = random.Random(seed)
    examples: list[dict[str, list[int]]] = []

    for triple in harmful_triples:
        if rng.random() < augment_prob:
            k = rng.randint(1, max_prefix_tokens)
            example = build_example(
                tokenizer,
                prompt=triple["prompt"],
                target=triple["refusal"],
                max_seq_length=max_seq_length,
                harmful_prefix=triple["harmful"],
                num_prefix_tokens=k,
            )
        else:
            example = build_example(tokenizer, triple["prompt"], triple["refusal"], max_seq_length)
        if example is not None:
            examples.append(example)

    for pair in benign_pairs:
        example = build_example(tokenizer, pair["prompt"], pair["response"], max_seq_length)
        if example is not None:
            examples.append(example)

    rng.shuffle(examples)
    return datasets.Dataset.from_list(examples)
