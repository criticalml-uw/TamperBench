# Running SN-Tune / RSN-Tune on Llama-2-7B-Chat and Mistral-7B-Instruct-v0.2

Wrapper scripts for running the original Safety-Neuron codebase on specific models,
deployable on a k8s cluster.

## Overview

The pipeline has two stages per model:
1. **Neuron detection** — identifies safety neurons (from harmful prompts) and
   foundation neurons (from Wikipedia) using the custom modeling files
2. **Neuron-specific training** — SN-Tune (safety neurons only) and RSN-Tune
   (safety neurons minus foundation neurons) using gradient masking

## Modifications from the original codebase

The original scripts (`neuron_detection/neuron_detection.py`,
`neuron_enhancement/train_neuron.py`) have hardcoded model names, placeholder paths
(`"xxxxxx"`), and depend on replacing entire files in the installed `transformers`
package. The custom files in the repo span different transformers versions
(~4.28, ~4.38), making them mutually incompatible with any single install.

### What these scripts change

**`detect_neurons.py`** — Parameterized replacement for `neuron_detection.py`:
- Accepts `--model`, `--corpus`, `--num-samples`, `--output` as CLI arguments
- Calls `model(input_ids=..., early_exit_layers=...)` directly instead of
  `model.generate()`, avoiding the need to patch `generation/utils.py` (which is
  from transformers ~4.28 and incompatible with modern versions)
- Only the model-specific files (`modeling_llama.py`, `modeling_mistral.py`) are
  patched into the installed transformers for detection

**`train_neurons.py`** — Parameterized replacement for `train_neuron.py`:
- Accepts `--model`, `--neuron-file`, `--data-file`, `--output-dir`, and
  optionally `--foundation-neuron-file` (enables RSN-Tune) as CLI arguments
- Implements gradient masking as a `SFTTrainer` subclass (`NeuronMaskingSFTTrainer`)
  with a custom `training_step()`, avoiding the need to patch `trainer.py` (which
  is from transformers ~4.28 and imports modules that no longer exist)
- The gradient masking logic is a faithful copy of the original `trainer.py` lines
  2019-2097: it truncates neuron sets to top 100 per layer, divides K/V indices
  by 4 (GQA kv_repeat), and zeros gradients for non-safety neurons after each
  backward pass

**`prepare_data.py`** — Downloads and formats the three required datasets:
- `AlignmentResearch/AdvBench` `content` column -> `harmful_behaviors.txt`
  (note: the `instructions` column is empty; the actual prompts are in `content`)
- `abhayesian/circuit-breakers-dataset` `prompt`/`chosen` columns ->
  `circuit_breakers_train.json` (JSON-lines with `original_question`/`response`)
- `wikimedia/wikipedia` (en, streaming) -> `wikipedia_en.txt`

**`run_pipeline.sh`** — Orchestrates the full pipeline:
- Installs `transformers==4.44.0 trl==0.8.6 peft==0.10.0 accelerate==0.33.0`
  (the one version combination where detection custom files, training, and all
  dependencies are mutually compatible)
- Patches modeling files for detection, restores them for training
- Runs detection (safety + foundation neurons), then SN-Tune and RSN-Tune

### Known issues / caveats

- **Llama-2-7B FFN top-K**: The custom `modeling_llama.py` uses
  `top_number_ffn=12000`, but Llama-2-7B has `intermediate_size=11008`. This means
  all FFN neurons are selected per prompt. The intersection across 520 prompts
  still filters this down, but the detected FFN neurons may be less selective than
  intended. The Mistral custom code uses `top_number_ffn=2000`.

- **GQA kv_repeat=4 hardcoded**: The gradient masking divides K/V attention
  indices by 4 (matching Llama-3-8B and Mistral-7B GQA with 8 KV heads). For
  Llama-2-7B-Chat which uses full MHA (32 KV heads), the divisor should be 1.
  This is preserved from the original code for faithful reproduction.

## Dependency versions

| Package | Version | Reason |
|---------|---------|--------|
| transformers | 4.44.0 | Needs `SlidingWindowCache` (4.42+), `MistralConfig.head_dim` (4.43+), no `non_blocking` Accelerator bug (fixed in 4.44) |
| trl | 0.8.6 | Needs to not import `top_k_top_p_filtering` (removed in transformers 4.40) |
| peft | 0.10.0 | Needs to not import `HybridCache` (not in transformers 4.44) |
| accelerate | 0.33.0 | Compatible with transformers 4.44 Trainer |

## Usage

### Local

```bash
# Prepare data
python scripts/prepare_data.py --data-dir ./data

# Run full pipeline for one model
bash scripts/run_pipeline.sh meta-llama/Llama-2-7b-chat-hf ./output/llama2
bash scripts/run_pipeline.sh mistralai/Mistral-7B-Instruct-v0.2 ./output/mistral
```

### K8s (u-tamperbench namespace)

Scripts are staged on the PVC at `/tamperbench_data/safety_neuron/scripts/`.

```bash
kubectl create -n u-tamperbench -f scripts/k8s_llama2.yaml
kubectl create -n u-tamperbench -f scripts/k8s_mistral.yaml
```

### Outputs

```
<output_dir>/
  safety_neurons.txt          # detected safety neurons (5-line format)
  foundation_neurons.txt      # detected foundation neurons
  sn_tune/final_model/        # SN-Tuned model weights + tokenizer
  rsn_tune/final_model/       # RSN-Tuned model weights + tokenizer
```
