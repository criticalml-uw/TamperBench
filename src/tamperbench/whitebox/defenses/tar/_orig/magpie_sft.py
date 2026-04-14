"""Post-TAR recovery SFT on Magpie-Align.

The TAR paper (Tamirisa et al. 2024, Appendix) notes that for the harmful
request refusal setting, an additional 100 steps of SFT on Magpie-Align is
needed after TAR training to recover benign capabilities.  This script
implements that step.

Invoked as a subprocess via ``accelerate launch`` from defense.py.
"""

import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import schedulefree
import torch
import wandb
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.dataloaders import get_magpie_dataloaders
from modules.utils import fix_seed


def _get_decoder_layer_class(model: torch.nn.Module) -> type:
    """Auto-detect the decoder layer class (same as tar_entry.py)."""
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name.endswith("DecoderLayer"):
            return type(module)
    raise RuntimeError(
        f"Could not auto-detect a DecoderLayer class in {type(model).__name__}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_data_size", type=int, default=40000)
    args = parser.parse_args()

    fix_seed()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    decoder_layer_cls = _get_decoder_layer_class(model)
    print(f"[Magpie SFT] FSDP wrapping: {decoder_layer_cls.__name__}")

    auto_wrap_policy = functools.partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda module: isinstance(module, decoder_layer_cls),
    )
    fsdp_plugin = FullyShardedDataParallelPlugin(auto_wrap_policy=auto_wrap_policy)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=fsdp_plugin,
    )

    if accelerator.is_main_process:
        wandb.init(mode="disabled")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = accelerator.prepare_model(model)
    _, magpie_train_dataloader = get_magpie_dataloaders(tokenizer, args, cutoff_len=1024)
    magpie_train_dataloader = accelerator.prepare(magpie_train_dataloader)
    magpie_iterator = iter(magpie_train_dataloader)

    model.config.use_cache = False
    model.train()

    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=args.lr, warmup_steps=args.warmup_steps
    )
    optimizer = accelerator.prepare(optimizer)
    optimizer.train()

    accelerator.print(f"[Magpie SFT] Starting {args.max_steps}-step recovery SFT")

    for step in range(args.max_steps):
        try:
            batch = next(magpie_iterator)
        except StopIteration:
            magpie_iterator = iter(magpie_train_dataloader)
            batch = next(magpie_iterator)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        loss = outputs.loss / args.gradient_accumulation_steps
        accelerator.backward(loss)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad(set_to_none=True)

        if accelerator.is_main_process and (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{args.max_steps}, loss={outputs.loss.item():.4f}")

    # Final optimizer step for any remaining accumulated gradients
    if args.max_steps % args.gradient_accumulation_steps != 0:
        optimizer.step()
        model.zero_grad(set_to_none=True)

    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)

    accelerator.print(f"[Magpie SFT] Done. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
