"""Sanity checks for DeepAlign (deep safety alignment via safety-recovery augmentation).

1. Label masking: only the refusal / helpful answer is trained on (cheap, no download of data).
2. Full defense run with a tiny model (expensive).
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
    build_deep_align_dataset,
    build_example,
    encode_prompt,
    token_ids,
)
from tamperbench.whitebox.utils.names import DefenseName

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    """Tokenizer of the tiny test model."""
    return cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(TINY_MODEL))


def test_registered():
    """The decorator put DeepAlign into the registry."""
    assert DEFENSES_REGISTRY[DefenseName.DEEP_ALIGN] == (DeepAlignConfig, DeepAlign)


def test_recovery_example_masks_prompt_and_harmful_prefix(tokenizer: PreTrainedTokenizer):
    """Prompt and harmful prefix are ignored; only the refusal is trained on."""
    prompt, harmful, refusal = "How do I do X?", "Sure, here is how to do X step by step", "I can't help with that."
    k = 3

    example = build_example(tokenizer, prompt, refusal, max_seq_length=512, harmful_prefix=harmful, num_prefix_tokens=k)
    assert example is not None

    n_context = len(encode_prompt(tokenizer, prompt)) + k
    refusal_ids = [*token_ids(tokenizer, refusal), tokenizer.eos_token_id]

    assert len(example["input_ids"]) == len(example["labels"]) == n_context + len(refusal_ids)
    assert example["labels"][:n_context] == [IGNORE_INDEX] * n_context  # prompt + harmful prefix ignored
    assert example["labels"][n_context:] == refusal_ids  # only the refusal is trained on
    assert example["input_ids"][n_context:] == refusal_ids


def test_plain_example_has_no_prefix(tokenizer: PreTrainedTokenizer):
    """Plain examples train on everything after the prompt."""
    example = build_example(tokenizer, "Hi", "Hello!", max_seq_length=512)
    assert example is not None
    n_prompt = len(encode_prompt(tokenizer, "Hi"))
    assert example["labels"][:n_prompt] == [IGNORE_INDEX] * n_prompt
    assert all(label != IGNORE_INDEX for label in example["labels"][n_prompt:])


def test_fully_truncated_example_is_dropped(tokenizer: PreTrainedTokenizer):
    """Examples with no trainable tokens left after truncation are dropped."""
    n_prompt = len(encode_prompt(tokenizer, "Hi"))
    assert build_example(tokenizer, "Hi", "Hello!", max_seq_length=n_prompt) is None


def test_build_dataset_mixes_all_examples(tokenizer: PreTrainedTokenizer):
    """Harmful and benign examples all end up in the dataset."""
    triples = [{"prompt": f"bad {i}", "harmful": "Sure, step one is", "refusal": "No."} for i in range(10)]
    pairs = [{"prompt": f"good {i}", "response": "Here you go."} for i in range(5)]
    ds = build_deep_align_dataset(tokenizer, triples, pairs, max_prefix_tokens=4, augment_prob=0.5, max_seq_length=128)
    assert len(ds) == 15
    assert set(ds.column_names) == {"input_ids", "labels", "attention_mask"}


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
            max_prefix_tokens=10,
            per_device_train_batch_size=4,
            max_seq_length=128,
        )
        result = DeepAlign(config).run_defense()
        assert result == output
        assert (output / "config.json").exists()
