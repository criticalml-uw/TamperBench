"""Sanity checks for DeepAlign (deep safety alignment via safety-recovery augmentation).

1. Label masking: only the refusal / helpful answer is trained on (cheap, no download of data).
2. Per-batch re-drawing of the harmful prefix length k (cheap).
3. Full defense run with a tiny model (expensive).
"""

import tempfile
from pathlib import Path
from typing import cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from tamperbench.whitebox.defenses import DEFENSES_REGISTRY
from tamperbench.whitebox.defenses.deep_align import DeepAlign, DeepAlignConfig
from tamperbench.whitebox.defenses.deep_align.data import (
    IGNORE_INDEX,
    SafetyRecoveryCollator,
    build_example,
    encode_prompt,
    token_ids,
    tokenize_examples,
)
from tamperbench.whitebox.utils.names import DefenseName

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    """Tokenizer of the tiny test model."""
    tok = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(TINY_MODEL, padding_side="right"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_registered():
    """The decorator put DeepAlign into the registry."""
    assert DEFENSES_REGISTRY[DefenseName.DEEP_ALIGN] == (DeepAlignConfig, DeepAlign)


def test_recovery_example_masks_prompt_and_harmful_prefix(tokenizer: PreTrainedTokenizer):
    """Prompt and harmful prefix are ignored; only the refusal is trained on."""
    prompt_ids = encode_prompt(tokenizer, "How do I do X?")
    prefix_ids = token_ids(tokenizer, "Sure, here is how to do X step by step")[:3]
    eos_id = tokenizer.eos_token_id
    assert isinstance(eos_id, int)
    refusal_ids = [*token_ids(tokenizer, "I can't help with that."), eos_id]

    example = build_example(prompt_ids, prefix_ids, refusal_ids, max_seq_length=512)
    n_context = len(prompt_ids) + 3

    assert len(example["input_ids"]) == len(example["labels"]) == n_context + len(refusal_ids)
    assert example["labels"][:n_context] == [IGNORE_INDEX] * n_context  # prompt + harmful prefix ignored
    assert example["labels"][n_context:] == refusal_ids  # only the refusal is trained on
    assert example["input_ids"][n_context:] == refusal_ids


def test_long_prefix_is_shortened_so_a_target_token_survives(tokenizer: PreTrainedTokenizer):
    """Truncation never leaves an example with nothing to train on."""
    prompt_ids = encode_prompt(tokenizer, "Hi")
    example = build_example(prompt_ids, list(range(100)), [7, 8, 9], max_seq_length=len(prompt_ids) + 5)
    assert len(example["input_ids"]) == len(prompt_ids) + 5
    assert example["labels"][-1] == 7


def test_unfinished_benign_response_has_no_eos(tokenizer: PreTrainedTokenizer):
    """A distilled answer cut off by the length cap is not taught to end with EOS."""
    pairs = [
        {"prompt": "a", "response": "done.", "finished": True},
        {"prompt": "b", "response": "and then", "finished": False},
    ]
    ds = tokenize_examples(tokenizer, [], pairs, max_prefix_tokens=4, max_seq_length=128)
    assert ds[0]["target_ids"][-1] == tokenizer.eos_token_id
    assert ds[1]["target_ids"][-1] != tokenizer.eos_token_id


def test_collator_redraws_prefix_length_each_batch(tokenizer: PreTrainedTokenizer):
    """The same harmful row gets different k across batches; benign rows never get a prefix."""
    triples = [{"prompt": "bad", "harmful": " ".join(["word"] * 50), "refusal": "No."}]
    pairs = [{"prompt": "good", "response": "Here you go."}]
    ds = tokenize_examples(tokenizer, triples, pairs, max_prefix_tokens=20, max_seq_length=256)
    harmful_row, benign_row = ds[0], ds[1]
    collator = SafetyRecoveryCollator(tokenizer, max_prefix_tokens=20, augment_prob=0.5, max_seq_length=256)

    n_prompt = len(harmful_row["prompt_ids"])
    n_target = len(harmful_row["target_ids"])
    prefix_lengths = set()
    for _ in range(200):
        batch = collator([harmful_row, benign_row])
        k = int(batch["attention_mask"][0].sum()) - n_prompt - n_target
        prefix_lengths.add(k)
        # Harmful row: prompt + prefix masked, refusal trained on.
        assert int((batch["labels"][0][: n_prompt + k + n_target] == IGNORE_INDEX).sum()) == n_prompt + k
        # Benign row: never gets a harmful prefix.
        assert int(batch["attention_mask"][1].sum()) == len(benign_row["prompt_ids"]) + len(benign_row["target_ids"])

    assert 0 in prefix_lengths  # plain refusals happen
    assert len(prefix_lengths) > 5  # many different k values
    assert max(prefix_lengths) <= 20


@pytest.mark.expensive
def test_defense_training():
    """End-to-end run on a tiny model produces a checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "deep_align"
        config = DeepAlignConfig(
            input_checkpoint_path=Path(TINY_MODEL),
            output_checkpoint_path=output,
            num_harmful_samples=8,
            num_benign_samples=8,
            distill_benign_responses=False,
            max_prefix_tokens=10,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            max_seq_length=128,
        )
        result = DeepAlign(config).run_defense()
        assert result == output
        assert (output / "config.json").exists()
