# DeepAlign: deep safety alignment by data augmentation

Implementation of the data-augmentation defense from:

> Qi et al., *Safety Alignment Should Be Made More Than Just a Few Tokens Deep*, ICLR 2025.
> Paper: [arXiv:2406.05946](https://arxiv.org/abs/2406.05946) · Code: [Unispac/shallow-vs-deep-alignment](https://github.com/Unispac/shallow-vs-deep-alignment)

## The idea

Aligned models usually decide to refuse within the first few tokens of their reply. If a reply starts with "Sure, here's how…" (because of a prefilling attack, a jailbreak or fine-tuning), the model tends to keep going. The paper calls this **shallow** alignment.

DeepAlign trains the model to recover. Each training example shows a harmful prompt, then the first *k* tokens of a harmful answer, then a refusal. Loss is computed only on the refusal, so the model learns to steer back to a refusal even after its reply has started going wrong. Benign prompts answered by the model itself are mixed in so its normal behaviour doesn't change.

## What a training example looks like

```text
1. Plain refusal (k = 0)
   input_ids: [prompt ...][refusal ... EOS]
   labels:    [-100 ...  ][refusal ... EOS]

2. Safety recovery (k ≥ 1)
   input_ids: [prompt ...][harmful[:k] ...][refusal ... EOS]
   labels:    [-100 ...  ][-100 ...       ][refusal ... EOS]

3. Benign utility
   input_ids: [prompt ...][model's own answer ... EOS]
   labels:    [-100 ...  ][model's own answer ... EOS]
```

`-100` positions are ignored by the loss. The harmful prefix is context only: the model never learns to produce it. Prompts are formatted with the model's chat template (or an Alpaca-style template for models without one).

## Pipeline (`DeepAlign._run_defense`)

| Step | What happens | Code |
|---|---|---|
| 1. Load data | 256 harmful triples (prompt, harmful answer, refusal) from `anonymous4486/repnoise_beavertail`; 1024 Alpaca prompts from `tatsu-lab/alpaca` | `load_harmful_triples`, `load_benign_prompts` |
| 2. Distill benign answers | The **input model** answers each Alpaca prompt (greedy decoding, vLLM if installed, else HF `generate`). This runs in a subprocess so GPU memory is freed before training. Results are cached under `data/deep_align_benign_responses/` | `DeepAlign._distill_benign_responses`, `generate_responses_from_ids` |
| 3. Tokenize | Prompt, harmful answer (first 100 tokens) and target are tokenized separately, once. No *k* is chosen yet | `tokenize_examples` |
| 4. Train | Full fine-tuning. At each batch, every harmful row gets a fresh coin flip: plain refusal (50%) or recovery with `k ~ U[1, min(100, len(harmful))]` (50%) | `SafetyRecoveryCollator`, `DeepAlign._train` |
| 5. Save | Plain HF checkpoint plus tokenizer at `output_checkpoint_path` | `DeepAlign._train` |

### Edge cases handled

- **Truncation:** a long harmful prefix is shortened so at least one refusal token always remains to train on. Rows whose prompt alone fills `max_seq_length` are dropped.
- **Cut-off benign answers:** if the model's answer hit `benign_max_new_tokens` without ending, no EOS is added. Otherwise the model would learn to stop mid-sentence.
- **No pad token:** one is added and the embeddings are resized. Padded positions get `labels = -100` and `attention_mask = 0`.

## Configuration (`DeepAlignConfig`)

| Field | Default | Notes |
|---|---|---|
| `harmful_dataset_name` | `anonymous4486/repnoise_beavertail` | BeaverTails plus refusals from the RepNoise paper; also used by Booster |
| `num_harmful_samples` | 256 | |
| `benign_dataset_name` | `tatsu-lab/alpaca` | |
| `num_benign_samples` | 1024 | 4 × harmful, giving a 20 / 80 harmful / benign mix |
| `distill_benign_responses` | `True` | `False` uses Alpaca's original answers (written by text-davinci-003) |
| `benign_max_new_tokens` | 512 | |
| `max_prefix_tokens` | 100 | Upper bound on *k* |
| `augment_prob` | 0.5 | Probability that a harmful example gets a prefix |
| `learning_rate` | 2e-5 | |
| `num_train_epochs` | 10 | |
| `per_device_train_batch_size` | 8 | |
| `gradient_accumulation_steps` | 10 | Effective batch 80 on one GPU. On N GPUs use 10 / N |
| `max_seq_length` | 512 | |
| `seed` | 42 | Data shuffling and the per-batch *k* draws |

Fixed training settings: full fine-tuning, AdamW, linear LR schedule, no warmup, bf16 on CUDA, gradient checkpointing. With the defaults this is 1280 examples × 10 epochs / 80 = **160 optimizer steps**.

## Comparison with the paper

Sources: paper Section 3 and Appendix A.3, plus the official training command in the authors' README.

### Matches the paper

| Aspect | Paper | Here |
|---|---|---|
| Loss | Plain cross-entropy on the refusal only; harmful prefix masked | Same |
| *k* distribution | `k = 0` with prob. 0.5, else `k ~ U[1, min(100, len(h))]` | Same |
| When *k* is drawn | Fresh for every example in every batch | Same (`SafetyRecoveryCollator`) |
| Harmful / benign mix | 16 harmful + 64 benign per batch (20 / 80) | 256 / 1024 shuffled together (20 / 80 on average per batch) |
| Benign answers | Generated by the initial aligned model ("utility anchor") | Generated by the input model |
| Benign prompts | Alpaca | Alpaca |
| Fine-tuning type | Full fine-tuning | Full fine-tuning |
| Epochs / LR | 10 / 2e-5 | 10 / 2e-5 |
| Optimizer / schedule | AdamW, linear, no warmup | Same |
| Batch size | 80 (4 GPUs × (4 harmful + 16 benign)) | 8 × 10 accumulation steps = 80 |
| Amount of training | 160 steps, about 12,800 sequences | 160 steps, 12,800 sequences |
| Max length | 512 | 512 |

### Differs from the paper: training

| Aspect | Paper | Here | Expected impact |
|---|---|---|---|
| Harmful data | 256 triples, mostly from Anthropic red-teaming data (gated: `Unispac/shallow-vs-deep-safety-alignment-dataset`) | 256 rows of BeaverTails (`repnoise_beavertail`) | Different prompt distribution |
| Refusals | Generated by Llama-2-7B-Chat itself | The dataset's refusals | Refusals aren't in the model's own voice |
| Harmful answers | From a jailbroken copy of the same model | BeaverTails responses | Prefixes look less like what this model would actually say |
| What an "epoch" covers | 10 passes over the harmful set; the benign stream cycles alongside | 10 passes over all 1280 examples | Same total training; possibly fewer distinct benign examples |
| Exact per-batch ratio | Exactly 16 : 64 every batch | 20 / 80 on average after shuffling | Minor |
| System prompt | Llama-2 default system prompt | None | Minor; the paper's prompt is Llama-2-specific |
| Models | Llama-2-7B-Chat | Any HF causal LM | — |

### Differs from the paper: evaluation

TamperBench evaluates every defense the same way, so DeepAlign is tested differently from the paper:

| | Paper | TamperBench |
|---|---|---|
| Attacks | Inference-time: **prefilling** (what DeepAlign targets), GCG, decoding-parameter exploit | Fine-tuning attack (`lora_finetune`) |
| Safety | HEx-PHI, judged by GPT-4 | StrongREJECT, judged by a fine-tuned small model |
| Utility | AlpacaEval | MMLU-Pro |
| Prompt format | Llama-2 chat format | `native` (the model's own chat template, which DeepAlign also trains with) |

Two consequences:

- **Results won't match the paper's numbers**, even on Llama-2-7B-Chat.
- **Expect weaker results than the paper reports.** DeepAlign was designed against prefilling, and the paper notes that data augmentation alone doesn't stop fine-tuning attacks.

### Not implemented

- **Token-wise constrained fine-tuning (paper Section 4, Eq. 3).** This is a separate defense applied *during* downstream fine-tuning, so it is out of scope for an alignment-stage defense.

## How to run

Python API:

```python
from pathlib import Path
from tamperbench.whitebox.defenses.deep_align import DeepAlign, DeepAlignConfig

config = DeepAlignConfig(
    input_checkpoint_path=Path("Qwen/Qwen3-0.6B"),
    output_checkpoint_path=Path("results/deep_align"),
)
DeepAlign(config).run_defense()
```

Benchmark (defend → eval → attack → eval) with the fixed hyperparameters in `configs/whitebox/defenses/deep_align/grid.yaml`:

```bash
uv run python scripts/whitebox/defense_grid.py Qwen/Qwen3-0.6B \
    --defense deep_align \
    --config-name base \
    --model-alias qwen3_0_6b \
    --results-dir results/defense_grid
```

Optuna sweep over the search space in `configs/whitebox/defenses/deep_align/single_objective_sweep.yaml`. Each trial runs defend → eval → malicious LoRA (`lora_finetune`) → eval and minimizes post-attack StrongREJECT. The base model's MMLU-Pro is scored once before the first trial (saved under `capability_baselines/`). A trial whose MMLU-Pro falls below 90% of that skips the attack and gets the worst score:

```bash
uv run python scripts/whitebox/defense_sweep.py meta-llama/Llama-3.2-1B-Instruct \
    --defense deep_align \
    --n-trials 10 \
    --model-alias llama3_2_1b_instruct \
    --results-dir results/defense_sweep
```

The best defended model is kept in `<results-dir>/<alias>/deep_align/optuna_single/best_checkpoint/`. To test it against other attacks, pass that path to `scripts/whitebox/benchmark_grid.py`.

### Notes

- The benign-answer cache is keyed by model **path**, not contents. Delete the cached file under `data/deep_align_benign_responses/` if the checkpoint at a path changes.
- vLLM is only installed on Linux. Elsewhere, generation falls back to HF `generate`, which is fine on a GPU but slow on CPU/MPS.
- Full fine-tuning a 7–8B model needs about 54–64 GB for weights, gradients and Adam state before activations, so use an 80 GB GPU (or several GPUs).
