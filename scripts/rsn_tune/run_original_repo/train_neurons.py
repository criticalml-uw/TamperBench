"""Parameterized SN-Tune / RSN-Tune training.

Implements the gradient masking from the original Safety-Neuron trainer.py
as a Trainer subclass, avoiding the need to patch the installed transformers.

Usage (SN-Tune):
    python scripts/train_neurons.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --neuron-file output/safety_neurons.txt \
        --data-file data/circuit_breakers_train.json \
        --output-dir output/sn_tune

Usage (RSN-Tune):
    python scripts/train_neurons.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --neuron-file output/safety_neurons.txt \
        --foundation-neuron-file output/foundation_neurons.txt \
        --data-file data/circuit_breakers_train.json \
        --output-dir output/rsn_tune
"""

import argparse
import copy
import itertools
import os
import re

import torch
from datasets import load_dataset
from peft import prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def retrive_neuron(filename):
    """Load neuron activation dicts from file (5 lines: fwd_up, fwd_down, q, k, v)."""
    activate_neuron = []
    with open(filename) as file:
        for line in file:
            neuron = eval(line.strip())
            activate_neuron.append(neuron)
    return activate_neuron


def deduplicate(neuron_target, neuron_delete, num_layers=32):
    """RSN-Tune: subtract foundation neurons from safety neurons."""
    index_keys = list(range(num_layers))
    for key in index_keys:
        for i in range(5):
            neuron_target[i][key] = neuron_target[i][key] - neuron_delete[i][key]
    return neuron_target


def build_neuron_masks(activate_neuron, num_layers=32):
    """Pre-compute the neuron masks from the original trainer.py logic.

    This replicates lines 2019-2062 of the custom trainer.py:
    - Splits layers into understanding (0-7), generation (last 4), reasoning (rest)
    - Truncates each neuron set to top 100 per layer
    - Divides attn_k and attn_v indices by 4 (GQA kv_repeat for Llama-3/Mistral)
    - Layers not in any group get empty neuron sets
    """
    index_keys = list(range(num_layers))
    index_keys_under = list(range(8))
    index_keys_gen = [num_layers - 1 - i for i in range(4)]
    index_keys_reason = [i for i in index_keys if i not in index_keys_under and i not in index_keys_gen]

    activate_fwd_up = copy.deepcopy(activate_neuron[0])
    activate_fwd_down = copy.deepcopy(activate_neuron[1])
    attn_q = copy.deepcopy(activate_neuron[2])
    attn_k = copy.deepcopy(activate_neuron[3])
    attn_v = copy.deepcopy(activate_neuron[4])

    # Original code divides k,v by 4 (GQA kv_repeat)
    attn_k = {key: {num // 4 for num in value} for key, value in attn_k.items()}
    attn_v = {key: {num // 4 for num in value} for key, value in attn_v.items()}

    for idx in index_keys:
        if idx in index_keys_under or idx in index_keys_reason or idx in index_keys_gen:
            activate_fwd_up[idx] = set(itertools.islice(activate_fwd_up[idx], 100))
            activate_fwd_down[idx] = set(itertools.islice(activate_fwd_down[idx], 100))
            attn_q[idx] = set(itertools.islice(attn_q[idx], 100))
            attn_k[idx] = set(itertools.islice(attn_k[idx], 100))
            attn_v[idx] = set(itertools.islice(attn_v[idx], 100))
        else:
            activate_fwd_up[idx] = []
            activate_fwd_down[idx] = []
            attn_q[idx] = []
            attn_k[idx] = []
            attn_v[idx] = []

    return activate_fwd_up, activate_fwd_down, attn_q, attn_k, attn_v


def apply_gradient_mask(model, activate_fwd_up, activate_fwd_down, attn_q, attn_k, attn_v):
    """Zero out gradients for non-safety neurons. Replicates lines 2065-2097."""
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            layer = int(match.group(1))
            if "attn.q_proj" in name:
                tune_index = attn_q[layer]
                mask = torch.ones(param.size(0), dtype=torch.bool)
                mask[list(tune_index)] = False
                param.grad[mask] = 0
            elif "attn.k_proj" in name:
                tune_index = attn_k[layer]
                mask = torch.ones(param.size(0), dtype=torch.bool)
                mask[list(tune_index)] = False
                param.grad[mask] = 0
            elif "attn.v_proj" in name:
                tune_index = attn_v[layer]
                mask = torch.ones(param.size(0), dtype=torch.bool)
                mask[list(tune_index)] = False
                param.grad[mask] = 0
            elif "up_proj" in name:
                tune_index = activate_fwd_up[layer]
                mask = torch.ones(param.size(0), dtype=torch.bool)
                mask[list(tune_index)] = False
                param.grad[mask] = 0
            elif "down_proj" in name:
                tune_index = activate_fwd_down[layer]
                mask = torch.ones(param.size(1), dtype=torch.bool)
                mask[list(tune_index)] = False
                param.grad.T[mask] = 0
            else:
                param.grad.zero_()
        else:
            param.grad.zero_()


class NeuronMaskingSFTTrainer(SFTTrainer):
    """SFTTrainer that applies gradient masking after each backward pass."""

    def __init__(self, *args, neuron_masks=None, **kwargs):
        """Initialize with optional neuron masks for gradient zeroing."""
        super().__init__(*args, **kwargs)
        self.neuron_masks = neuron_masks

    def training_step(self, model, inputs):
        """Run a training step and zero gradients for non-safety neurons."""
        loss = super().training_step(model, inputs)
        if self.neuron_masks is not None:
            apply_gradient_mask(model, *self.neuron_masks)
        return loss


def formatting_prompts_func(example):
    """Format training examples as plain text: ``question. response``."""
    output_texts = []
    for i in range(len(example["original_question"])):
        text = f"{example['original_question'][i]}. {example['response'][i]}"
        output_texts.append(text)
    return output_texts


def main():
    """Run SN-Tune or RSN-Tune training."""
    parser = argparse.ArgumentParser(description="SN-Tune / RSN-Tune training.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--neuron-file", type=str, required=True)
    parser.add_argument("--foundation-neuron-file", type=str, default=None)
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training samples (paper uses 50)")
    args = parser.parse_args()

    mode = "RSN-Tune" if args.foundation_neuron_file else "SN-Tune"
    print(f"Running {mode} on {args.model}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load neurons
    activate_neuron = retrive_neuron(args.neuron_file)
    if args.foundation_neuron_file:
        foundation_neuron = retrive_neuron(args.foundation_neuron_file)
        activate_neuron = deduplicate(activate_neuron, foundation_neuron, args.num_layers)
        print("RSN-Tune: subtracted foundation neurons from safety neurons")

    # Build masks
    neuron_masks = build_neuron_masks(activate_neuron, args.num_layers)

    # Load dataset
    dataset = load_dataset("json", data_files=args.data_file, split="train", cache_dir=args.cache_dir)
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
    print(f"Using {len(dataset)} training samples")

    # Load model
    print(f"Loading model: {args.model}")
    base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    for name, param in tqdm(base_model.named_parameters(), desc="Enabling gradients"):
        param.requires_grad = True

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        save_steps=500,
        save_total_limit=0,
        logging_steps=10,
        output_dir=args.output_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
    )

    trainer = NeuronMaskingSFTTrainer(
        base_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=512,
        formatting_func=formatting_prompts_func,
        args=training_args,
        neuron_masks=neuron_masks,
    )

    trainer.train()

    # Save model
    model_output_dir = os.path.join(args.output_dir, "final_model")
    trainer.model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"{mode} model saved to {model_output_dir}")


if __name__ == "__main__":
    main()
