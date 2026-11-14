"""Parameterized neuron detection — replaces neuron_detection/neuron_detection.py.

This script detects activated neurons for a given model and corpus.
It must be run AFTER patching the installed transformers package with the
custom model files from neuron_detection/transformers/ (modeling_llama.py,
modeling_mistral.py). It does NOT require patching generation/utils.py —
it calls the model forward directly instead of using model.generate().

Usage:
    python scripts/detect_neurons.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --corpus data/harmful_behaviors.txt \
        --num-samples 520 \
        --output output/llama2_safety_neurons.txt
"""

import argparse
import os
import random

import torch
from tqdm import tqdm

random.seed(112)


@torch.no_grad()
def detect_single_prompt(model, tokenizer, prompt, candidate_premature_layers):
    """Run a single forward pass and extract neuron activations."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # The custom modeling files accept early_exit_layers and return neuron
    # activations as a 9-tuple: (logits_dict, outputs, fwd_up, fwd_down,
    # q, k, v, o, layer_keys)
    result = model(
        input_ids=inputs.input_ids,
        early_exit_layers=candidate_premature_layers,
    )
    (
        logits_dict,
        outputs,
        activate_keys_fwd_up,
        activate_keys_fwd_down,
        activate_keys_q,
        activate_keys_k,
        activate_keys_v,
        activate_keys_o,
        layer_keys,
    ) = result
    return (
        activate_keys_fwd_up,
        activate_keys_fwd_down,
        activate_keys_q,
        activate_keys_k,
        activate_keys_v,
    )


def find_common_neurons(activate_keys_list):
    """Find neurons activated across ALL prompts (intersection)."""
    common = {}
    if not activate_keys_list:
        return common
    for key in activate_keys_list[0].keys():
        if all(key in d for d in activate_keys_list):
            arrays = [d[key] for d in activate_keys_list]
            common[key] = set.intersection(*map(set, arrays))
    return common


def main():
    """Detect activated neurons for a model using the patched transformers."""
    parser = argparse.ArgumentParser(description="Detect activated neurons for a model.")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus text file (one line per prompt)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to use (default: all)")
    parser.add_argument("--output", type=str, required=True, help="Output path for neuron file")
    parser.add_argument("--num-layers", type=int, default=32, help="Number of model layers (default: 32)")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    # Read corpus
    with open(args.corpus) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if args.num_samples and args.num_samples < len(lines):
        lines = random.sample(lines, args.num_samples)
    print(f"Using {len(lines)} prompts from {args.corpus}")

    candidate_premature_layers = list(range(args.num_layers))

    sets_fwd_up = []
    sets_fwd_down = []
    sets_q = []
    sets_k = []
    sets_v = []
    error_count = 0

    for prompt in tqdm(lines, desc="Detecting neurons"):
        try:
            fwd_up, fwd_down, q, k, v = detect_single_prompt(model, tokenizer, prompt, candidate_premature_layers)
            sets_fwd_up.append(fwd_up)
            sets_fwd_down.append(fwd_down)
            sets_q.append(q)
            sets_k.append(k)
            sets_v.append(v)
        except Exception as e:
            error_count += 1
            print(f"Error ({error_count}): {e}")

    print(f"Completed with {error_count} errors out of {len(lines)} prompts")

    # Find intersection across all prompts
    common_fwd_up = find_common_neurons(sets_fwd_up)
    common_fwd_down = find_common_neurons(sets_fwd_down)
    common_q = find_common_neurons(sets_q)
    common_k = find_common_neurons(sets_k)
    common_v = find_common_neurons(sets_v)

    # Write output (same format as original code: 5 lines, one dict per line)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(str(common_fwd_up) + "\n")
        f.write(str(common_fwd_down) + "\n")
        f.write(str(common_q) + "\n")
        f.write(str(common_k) + "\n")
        f.write(str(common_v) + "\n")

    print(f"Neurons written to {args.output}")


if __name__ == "__main__":
    main()
