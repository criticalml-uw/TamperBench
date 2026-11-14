import argparse
import functools
import os
import sys
import random
from typing import Callable

# Ensure local modules (configs/, modules/) resolve when invoked as a subprocess.
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import schedulefree
import torch
import wandb
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from configs.config import SAVE_MODELS_DIR
from modules.dataloaders import (
    get_tar_dpo_dataloaders,
    get_tar_bio_dataloaders,
    get_tar_cyber_dataloaders,
)
from modules.training import random_mapping_training_loop, tar_training_loop
from modules.utils import fix_seed


def _get_decoder_layer_class(model: torch.nn.Module) -> type:
    """Auto-detect the decoder layer class from a transformer model.

    Walks the model's module tree looking for a class whose name ends with
    "DecoderLayer" (e.g. LlamaDecoderLayer, Qwen3DecoderLayer, etc.).  This is
    the layer that FSDP should wrap for sharding.

    To add support for a new architecture, no changes are needed here as long as
    the architecture follows the HuggingFace naming convention.  If the
    architecture uses an unusual name, add a suffix check for it in the loop.
    """
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name.endswith("DecoderLayer"):
            return type(module)
    raise RuntimeError(
        f"Could not auto-detect a DecoderLayer class in {type(model).__name__}. "
        "You may need to update _get_decoder_layer_class() in tar_entry.py."
    )


def finetune_no_trainer(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    output_dir: str = None,
    loop_type: Callable = tar_training_loop,
    dataloader_type: Callable = get_tar_bio_dataloaders,
    args: argparse.Namespace = None,
):
    # Load model and auto-detect FSDP wrap target
    model = AutoModelForCausalLM.from_pretrained(model_name)
    decoder_layer_cls = _get_decoder_layer_class(model)
    print(f"FSDP wrapping: {decoder_layer_cls.__name__}")

    auto_wrap_policy = functools.partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda module: isinstance(module, decoder_layer_cls),
    )
    FSDP_PLUGIN = FullyShardedDataParallelPlugin(
        auto_wrap_policy=auto_wrap_policy,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=FSDP_PLUGIN,
    )

    # Wandb logging
    if accelerator.is_main_process:
        wandb_mode = "online" if args.wandb else "disabled"
        if args.wandb:
            wandb.login()
        wandb.init(
            project=args.wandb_project_name,
            config=args,
            name="_".join(output_dir.split("/")),
            mode=wandb_mode,
        )
    accelerator.print("Beginning Training.")
    accelerator.free_memory()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # prepare model before optimizer: https://huggingface.co/blog/pytorch-fsdp
    model = accelerator.prepare_model(model)
    dataloaders = dataloader_type(tokenizer, accelerator, args=args, model=model)

    model.train()
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=args.lr, warmup_steps=args.warmup_steps
    )
    optimizer = accelerator.prepare(optimizer)
    accelerator.print(f"model, optimizers, dataloaders prepared")
    accelerator.print(f"output_dir: {output_dir}")

    # Calls either the TAR loop or random vectors loop
    model = loop_type(
        model,
        dataloaders,
        optimizer,
        accelerator,
        **vars(args),
    )
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    # Save tokenizer alongside model so the checkpoint is self-contained
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)


# Map the subject to the dataloader
DATALOADER_MAP = {
    "bio": get_tar_bio_dataloaders,
    "cyber": get_tar_cyber_dataloaders,
    "dpo_anthropic": get_tar_dpo_dataloaders,
}

# Map for training loops
TRAINING_CONFIG = {
    "random_mapping_trainer": random_mapping_training_loop,
    "tar_trainer": tar_training_loop,
}


def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_model_name", "-od", type=str, default="tar_model")
    parser.add_argument("--trainer_type", "-tt", type=str, default="tar_trainer")
    parser.add_argument("--max_data_size", "-mds", type=int, default=40000)
    parser.add_argument("--concept_data_split", "-cs", type=float, default=0.2)
    parser.add_argument("--lr", "-lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", "-bs", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", "-ga", type=int, default=8)
    parser.add_argument("--max_steps", "-ms", type=int, default=750)
    parser.add_argument("--inner_optimizer_warmup_steps", "-iws", type=int, default=20)
    parser.add_argument("--warmup_steps", "-ws", type=int, default=50)
    parser.add_argument("--expname", "-en", type=str, default="latest")
    parser.add_argument(
        "--base_model_name",
        "-bm",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--retain_model_name",
        "-rm",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--tar_inner_loop_steps", "-is", type=int, default=1)
    parser.add_argument("--tar_num_tasks_sampled", "-mnts", type=int, default=1)
    parser.add_argument("--retain_representations", "-rr", action="store_true")
    parser.add_argument(
        "--tar_tamper_resistance_loss_lower_bound", "-mlb", type=float, default=-11.76
    )
    parser.add_argument("--use_weighting_schedule", "-uws", action="store_true")
    parser.add_argument("--subject", "-st", type=str, default="bio-multi-dists")
    parser.add_argument("--tar_inner_loop_subsample", "-mils", type=int, default=1)
    parser.add_argument("--tar_adversary_batch_size", "-ilbs", type=int, default=1)
    parser.add_argument("--schedule_lambda", "-sl", type=float, default=0.5)
    parser.add_argument(
        "--tar_tamper_resistance_grad_scale", "-mgs", type=float, default=4.0
    )
    parser.add_argument("--tar_retain_scale", "-mrs", type=float, default=1.0)
    parser.add_argument(
        "--tar_tamper_resistance_loss_type", "-mlt", type=str, default="max_entropy"
    )
    parser.add_argument(
        "--adversary_dist_types",
        "-advs",
        type=str,
        default="pile-bio:0.33,camel-bio:0.33,retain_forget_switch:0.33",
    )
    parser.add_argument(
        "--switching_point_coeffs", "-spc", type=str, default="alpha:6.0,beta:3.0"
    )
    parser.add_argument(
        "--adversary_lr_schedulers", "-alrs", type=str, default="constant:1.0"
    )
    parser.add_argument(
        "--adversary_lr_samples", "-als", type=str, default="2e-6,2e-5,4e-5"
    )
    parser.add_argument("--wandb", "-wb", action="store_true")
    parser.add_argument("--unbounded", "-ub", action="store_true")
    parser.add_argument("--retain_same_base", "-rsb", action="store_true")
    parser.add_argument(
        "--wandb_project_name", "-wpn", type=str, default="tar_training"
    )
    args = parser.parse_args()
    fix_seed()
    finetune_no_trainer(
        model_name=args.base_model_name,
        output_dir=os.path.join(
            SAVE_MODELS_DIR, f"{args.new_model_name}_{args.expname}"
        ),
        loop_type=TRAINING_CONFIG[args.trainer_type],
        dataloader_type=DATALOADER_MAP[args.subject],
        args=args,
    )


if __name__ == "__main__":
    main()
