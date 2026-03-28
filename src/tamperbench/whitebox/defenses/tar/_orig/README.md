# Original TAR code (`_orig/`)

This directory contains the TAR training code from
[rishub-tamirisa/tamper-resistance](https://github.com/rishub-tamirisa/tamper-resistance),
copied mostly verbatim. The TamperBench facade in `../defense.py` invokes
`tar_entry.py` as a subprocess via `accelerate launch`.

## Changes from upstream

### `tar_entry.py` (was `tar.py` upstream)

1. **`sys.path.insert`** at the top so `configs/` and `modules/` resolve when
   invoked as a subprocess from an arbitrary working directory.

2. **Model-agnostic FSDP wrapping.** Replaced the hardcoded
   `LlamaDecoderLayer` FSDP wrap policy with `_get_decoder_layer_class()`,
   which walks the model's module tree and finds any class ending in
   `"DecoderLayer"`. This works for Llama, Qwen3, Mistral, Gemma, and any
   other HuggingFace model that follows the standard naming convention. If a
   future architecture uses an unusual name, add a suffix check for it in the
   loop inside `_get_decoder_layer_class()`.

3. **`AutoModelForCausalLM` instead of architecture-specific classes.** Removed
   `MODEL_MAP` / `TOKENIZER_MAP` / `--base` arg. The model is loaded with
   `AutoModelForCausalLM.from_pretrained()` and the tokenizer is loaded from
   the same checkpoint.

4. **Tokenizer saved with checkpoint.** Added `tokenizer.save_pretrained()` so
   the output checkpoint is self-contained for downstream inference (e.g. vllm).

5. **Conditional `wandb.login()`** -- only called when `--wandb` is passed, so
   runs without a wandb API key don't crash.

6. **Conditional `pad_token` assignment** -- `tokenizer.pad_token` is only set
   to `eos_token` when it's `None` (avoids overriding tokenizers that already
   have a distinct pad token, e.g. Qwen3).

### `modules/training.py`

7. **`optimizer.train()` for schedulefree.** Added `optimizer.train()` calls
   before the outer training loop in `tar_training_loop` and
   `random_mapping_training_loop`. Newer versions of `schedulefree` require
   explicit `.train()` / `.eval()` mode switching.

### `modules/dataloaders.py`

8. **Guarded `.strip(bos_token)` / `.strip(eos_token)` calls.** Some tokenizers
   (e.g. Qwen3) have `bos_token=None`; calling `.strip(None)` raises
   `TypeError`. Added `if tokenizer.bos_token:` guards in
   `get_pile_bio_retain_forget_heldout_datasets`, `get_cyber_datasets`, and
   `hh_rlhf_format`.

9. **`max_size` parameter for `get_magpie_datasets()`** -- allows limiting the
   magpie dataset size via `max_data_size` (useful for debug runs; the full
   dataset is ~98K examples).

## Adding support for new models

No code changes should be needed as long as:
- The model works with `AutoModelForCausalLM.from_pretrained()`
- The decoder layer class name ends with `"DecoderLayer"`
- The tokenizer loads from the same checkpoint path

If any of these don't hold, see the specific notes above for where to add
fallbacks.
