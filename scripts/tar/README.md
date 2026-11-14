# TAR Defense Reproduction (Harmful Request Refusal)

Reproduction of the harmful request refusal experiments from the TAR paper
(Tamirisa et al. 2024, "Tamper-Resistant Safeguards for Open-Weight LLMs",
ICLR 2025).

## Results

Model: Llama-3-8B-Instruct. 8x H100 80GB GPUs. TAR training: ~4.5 hours.

### Pre-attack (defended model quality)

| Metric       | Paper (base) | Ours (base) | Paper (TAR) | Ours (TAR) |
|--------------|--------------|-------------|-------------|------------|
| MT-Bench (↑) | 8.1          | 7.356       | 6.3         | 5.975      |
| MMLU-Pro (↑) | —            | 0.446       | —           | 0.314      |

### Post-attack (tamper resistance)

The paper reports HarmBench ASR; we use StrongREJECT (finetuned classifier).
Direct numeric comparison is not meaningful, but relative ranking is.

|                        | Defended | Undefended |
|------------------------|----------|------------|
| Avg StrongREJECT (↓)   | 0.717    | 0.760      |
| Avg MMLU-Pro Val (↑)   | 0.339    | 0.413      |

Per-adversary StrongREJECT (defended / undefended):

| Adversary   | Defended | Undefended |
|-------------|----------|------------|
| base        | 0.697    | 0.770      |
| adv1_warmup | 0.726    | 0.763      |
| adv3_bs16   | 0.676    | 0.771      |
| adv4_lr2e5  | 0.774    | 0.783      |
| adv5_lr4e5  | 0.711    | 0.712      |

The defended model shows modestly lower StrongREJECT. However, it's possible the
result is noise or is a result of weaker capabilities/coherence rather than
purely reduced harmfulness.

## Evaluation differences from the paper

### Safety metric: StrongREJECT vs HarmBench ASR

The paper uses **HarmBench ASR** (attack success rate on 1,528 harmful
behaviors, scored by a finetuned Llama-2-13B classifier). We use
**StrongREJECT** (finetuned classifier on a different prompt set). These
metrics measure overlapping but not identical things, so our numeric scores
are not directly comparable to the paper's 63.9% defended / 72.5% undefended
ASR.

To get directly comparable numbers, we'd need to evaluate with HarmBench's
classifier — see the "HarmBench integration" section below.

## Running

Full run (8x H100, ~4.5h TAR + ~1h SFT + ~4h evals/attacks):

```bash
python scripts/tar/test_tar.py meta-llama/Meta-Llama-3-8B-Instruct \
    --results-dir /path/to/results --num-gpus 8
```

Debug mode (1x GPU, Qwen3-0.6B, ~5 min end-to-end pipeline check):

```bash
python scripts/tar/test_tar.py --debug
```

Resume from existing defended checkpoint (skips TAR training):

```bash
python scripts/tar/test_tar.py meta-llama/Meta-Llama-3-8B-Instruct \
    --defended-checkpoint /path/to/defended_model \
    --results-dir /path/to/results --num-gpus 8
```
