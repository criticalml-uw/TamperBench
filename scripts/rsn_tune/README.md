# RSN-Tune experiments

Scripts for evaluating the RSN-Tune defense from
[Zhao et al. (2025) "Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron" (ICLR 2025)](https://openreview.net/forum?id=yR47RmND1m).

## Table 4 replication

`table4.py` replicates Table 4 (Section 4, "More Robust Efficient Safety Tuning")
from the paper. It evaluates whether RSN-Tune preserves model safety when the
model is subsequently fine-tuned on a benign downstream task (GSM8K).

### Conditions

| Condition | Description |
|-----------|-------------|
| Before    | Base instruction-tuned model, no fine-tuning |
| Original  | Fine-tuned on GSM8K (no defense) |
| SN-Tune   | SN-Tune defense, then fine-tuned on GSM8K |
| RSN-Tune  | RSN-Tune defense, then fine-tuned on GSM8K |

### Usage

```bash
# Run all four conditions:
python scripts/rsn_tune/table4.py meta-llama/Llama-2-7b-chat-hf

# Skip conditions that already completed:
python scripts/rsn_tune/table4.py meta-llama/Llama-2-7b-chat-hf \
    --skip-conditions before original

# Submit to SLURM (CAIS cluster):
bash scripts/rsn_tune/submit_table4.sh
bash scripts/rsn_tune/submit_table4.sh --model mistralai/Mistral-7B-Instruct-v0.2
bash scripts/rsn_tune/submit_table4.sh --dry-run
```

### Results: Llama-2-7B-Chat

Defense lr=1e-6 (paper value):

|                        | Before | Original | SN-Tune | RSN-Tune |
|------------------------|-------:|---------:|--------:|---------:|
| StrongREJECT (harmful) |  0.063 |    0.153 |   0.140 |    0.173 |
| &ensp;(post-defense)   |    --- |      --- |   0.069 |    0.076 |
| MMLU-Pro (capability)  |  0.232 |    0.177 |   0.177 |    0.175 |
| &ensp;(post-defense)   |    --- |      --- |   0.227 |    0.234 |

Earlier run with defense lr=2e-6 (original codebase value):

|                        | Before | Original | SN-Tune | RSN-Tune |
|------------------------|-------:|---------:|--------:|---------:|
| StrongREJECT (harmful) |  0.085 |    0.195 |   0.151 |    0.153 |
| &ensp;(post-defense)   |    --- |      --- |   0.099 |    0.099 |
| MMLU-Pro (capability)  |  0.230 |    0.179 |   0.191 |    0.173 |
| &ensp;(post-defense)   |    --- |      --- |   0.227 |    0.230 |

Observations:
- GSM8K benign fine-tuning degrades safety (Before -> Original), consistent with
  the paper's finding.
- We do not observe the paper's finding that RSN-Tune preserves safety better
  than SN-Tune after downstream fine-tuning.
- The "before" and "original" conditions (which don't use the defense and hence
  should be the same between both runs) show variance, and the SN-Tune and
  RSN-Tune scores fall within this noise band. We therefore do not have evidence
  that our implementation of RSN-Tune works as a defense.
- The paper's evaluation uses average ASR across four adversarial attack methods
  (Direct, GCG, AutoDAN, PAIR), which may be more sensitive to the defense
  differences than our StrongREJECT direct-prompting metric.

### Paper reference (Table 4)

|         | Before | Original | SN-Tune | RSN-Tune |
|---------|-------:|---------:|--------:|---------:|
| GSM8K   |   16.8 |     26.5 |    27.2 |     26.2 |
| Harmful |    0.0 |     41.0 |    38.0 |     26.0 |

(Harmful = average ASR across Direct Attack, GCG, AutoDAN, and PAIR on AdvBench.)

### Discrepancies vs. paper

1. **Safety metric.** The paper averages ASR across four adversarial methods
   (Direct, GCG, AutoDAN, PAIR) on AdvBench via HarmBench. We use StrongREJECT
   with direct prompting only. Absolute numbers are not comparable.
2. **GSM8K fine-tuning hyperparameters.** The paper does not specify LR, batch
   size, or optimizer for the GSM8K step. We use lr=2e-5, batch_size=8, AdamW,
   full-parameter fine-tuning on the full train set (7,473 samples).
3. **Capability metric.** The paper reports GSM8K accuracy. We report MMLU-Pro
   (val) instead.
