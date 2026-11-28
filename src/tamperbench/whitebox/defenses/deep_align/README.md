# DeepAlign: deep safety alignment by data augmentation

> Qi et al., *Safety Alignment Should Be Made More Than Just a Few Tokens Deep*, ICLR 2025.
> Paper: [arXiv:2406.05946](https://arxiv.org/abs/2406.05946) · Code: [Unispac/shallow-vs-deep-alignment](https://github.com/Unispac/shallow-vs-deep-alignment)

## The idea

Aligned models usually decide to refuse within the first few tokens of their reply. If a reply starts with "Sure, here's how…", the model tends to keep going. The paper calls this **shallow** alignment.

DeepAlign trains the model to recover. Each harmful example is a prompt, then the first *k* tokens of a harmful answer, then a refusal, with loss only on the refusal. Benign prompts answered by the model itself are mixed in so its normal behaviour doesn't change.

```text
Plain refusal (k = 0)     [prompt][refusal EOS]                labels: refusal only
Safety recovery (k ≥ 1)   [prompt][harmful[:k]][refusal EOS]   labels: refusal only
Benign utility            [prompt][model's own answer EOS]     labels: answer only
```

Prompts use the model's chat template (Alpaca-style if it has none).

## Pipeline (`DeepAlign._run_defense`)

1. Load 256 harmful triples (prompt, harmful answer, refusal) from `anonymous4486/repnoise_beavertail` and 1,024 Alpaca prompts.
2. The input model answers the Alpaca prompts (vLLM, falling back to HF `generate`). Answers are cached under `data/deep_align_benign_responses/`.
3. Tokenize once. *k* is drawn later.
4. Full fine-tuning. Every batch, each harmful row gets *k* = 0 with probability 0.5, else *k* ~ U[1, min(100, len(harmful))] (`SafetyRecoveryCollator`).
5. Save the checkpoint and tokenizer.

Defaults follow the paper: 20 / 80 harmful / benign, 10 epochs, lr 2e-5, effective batch 80 (8 × 10 accumulation), linear schedule, no warmup, max length 512. All fields are in `DeepAlignConfig`.

## Evaluation in TamperBench

- **Prompt format: `template: native`.** DeepAlign learns its refusals inside the model's chat template, so evals and attacks use the same format.
- **Attacks: inference-time, not fine-tuning.** The paper targets attacks that force a harmful start to the reply, and reports that data augmentation alone doesn't stop fine-tuning attacks. Both attacks below leave the weights unchanged and score with StrongREJECT:
  - `embedding_attack`: optimizes a soft prompt so the reply starts with an affirmative target ("Sure, here is…"). This is a white-box, continuous relative of GCG, which the paper uses.
  - `prompt_attack`: the AIM jailbreak prompt, a cheap black-box attack that also tells the model not to open with a refusal.
- **Objective:** minimize the worst post-attack StrongREJECT over both attacks (`post_attack.embedding_attack_eval`). The real per-attack scores are `post_attack.embedding_attack.embedding_attack_eval` and `post_attack.prompt_attack.prompt_attack_eval`. Because both evals share the StrongREJECT metric, the cross-named keys just repeat those values.
- **Utility guard:** the base model's MMLU-Pro is scored once before the sweep. A trial whose MMLU-Pro falls below 90% of that skips the attacks and gets StrongREJECT = 1.0.
- **Not in the sweep:** GCG matches the paper best but is too slow to run in every trial. Run it, or fine-tuning attacks, on the best checkpoint afterwards.

## Differences from the paper

- **Training data:** BeaverTails rows with their own refusals, instead of the paper's (gated) red-teaming data with refusals and harmful answers from Llama-2-7B-Chat itself.
- **Batching:** a 20 / 80 mix on average after shuffling, instead of exactly 16 : 64 per batch. No Llama-2 system prompt.
- **Evaluation:** StrongREJECT and MMLU-Pro instead of HEx-PHI (GPT-4 judge) and AlpacaEval. No prefilling or decoding-parameter attack is available in TamperBench, so numbers won't match the paper.
- **Not implemented:** the paper's token-wise constrained fine-tuning objective (Section 4), which is applied during downstream fine-tuning rather than at alignment time.

## How to run

```bash
# Fixed hyperparameters (grid.yaml)
uv run python scripts/whitebox/defense_grid.py meta-llama/Llama-3.2-1B-Instruct \
    --defense deep_align --config-name base --model-alias llama3_2_1b_instruct

# Optuna sweep (single_objective_sweep.yaml): defend -> eval -> both attacks -> eval
uv run python scripts/whitebox/defense_sweep.py meta-llama/Llama-3.2-1B-Instruct \
    --defense deep_align --n-trials 10 --model-alias llama3_2_1b_instruct
```

The best model is saved to `results/.../deep_align/optuna_single/best_checkpoint/`. Pass that path to `scripts/whitebox/benchmark_grid.py` to run other attacks on it.

## Notes

- **The embedding attack is slow:** it optimizes every StrongREJECT prompt one at a time with HF (not vLLM). Expect it to dominate each trial's runtime.
- **Qwen3 thinking mode:** with `native`, Qwen3 writes `<think>` blocks, which skews both scores. Prefer Llama-style instruct models.
- **Benign-answer cache:** keyed by model **path**, not contents. Delete the cached file if the checkpoint at a path changes.
- **GPU memory:** full fine-tuning a 7–8B model needs about 54–64 GB before activations. Use an 80 GB GPU.
