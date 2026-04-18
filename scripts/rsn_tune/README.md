# RSN-Tune experiments

Scripts for evaluating the RSN-Tune defense from
[Zhao et al. (2025) "Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron" (ICLR 2025)](https://openreview.net/forum?id=yR47RmND1m).

We did not successfully replicate the results from the paper though we do have
differences in our evaluation setup.

## Table 4 replication

`table4.py` replicates Table 4 (Section 4, "More Robust Efficient Safety Tuning")
from the paper. It evaluates whether RSN-Tune preserves model safety when the
model is subsequently fine-tuned on a benign downstream task (GSM8K).

### Conditions

| Condition  | Description                                                    |
|------------|----------------------------------------------------------------|
| Before     | Base instruction-tuned model, no fine-tuning                   |
| Undefended | Benign fine-tuning on GSM8K (no defense applied first)         |
| SN-Tune    | SN-Tune defense applied, then benign fine-tuning on GSM8K      |
| RSN-Tune   | RSN-Tune defense applied, then benign fine-tuning on GSM8K     |

"Fine-tuning" here refers to benign downstream task fine-tuning on GSM8K.  The
paper's finding is that even benign fine-tuning can degrade safety alignment,
and RSN-Tune is meant to make models more robust to this degradation.

"Paper" mode follows the paper's formulation (threshold-based detection,
separable norms, chat template, lr=1e-6). The paper has some differences from
the original codebase, so "Orig-code" mode follows the original codebase more
closely (top-K + intersection detection, L1 metrics, AdvBench for detection,
plain text, lr=2e-6). See the `match_original_code` flag in `RSNTuneConfig`.

### Usage

```bash
# Run all four conditions:
python scripts/rsn_tune/table4.py meta-llama/Llama-2-7b-chat-hf

# With original-code mode:
python scripts/rsn_tune/table4.py meta-llama/Llama-2-7b-chat-hf --original-code

# With a different fine-tuning config (lower LR):
python scripts/rsn_tune/table4.py meta-llama/Llama-2-7b-chat-hf --attack-config gsm8k_lr5e6

# With more defense training samples:
python scripts/rsn_tune/table4.py meta-llama/Llama-2-7b-chat-hf --defense-config-suffix _2k_train

# Skip conditions that already completed:
python scripts/rsn_tune/table4.py meta-llama/Llama-2-7b-chat-hf \
    --skip-conditions before undefended
```

### Results

StrongREJECT = StrongREJECT score (higher = more harmful).
MMLU-Pro = MMLU-Pro val accuracy (higher = better).
GSM8K fine-tuning uses effective batch size 128, 1 epoch, AdamW, constant LR
(following Qi et al. 2024), with lr=2e-5 unless noted.

#### Llama-2-7B-Chat, default defense (50 training samples), 5 seeds

|              | Before          | Undefended      | SN-Tune         | RSN-Tune        | SN-Orig         | RSN-Orig        |
|--------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| StrongREJECT | 0.076 ± 0.019   | 0.142 ± 0.019   | 0.134 ± 0.016   | 0.149 ± 0.019   | 0.144 ± 0.012   | 0.135 ± 0.017 |
| MMLU-Pro     | 0.230 ± 0.001   | 0.207 ± 0.009   | 0.205 ± 0.006   | 0.201 ± 0.010   | 0.206 ± 0.005   | 0.204 ± 0.008 |

All conditions fall within the same noise band. No signal that SN-Tune or
RSN-Tune does better than Undefended.

#### Llama-2-7B-Chat, 2000 defense training samples, 1 seed

|              | Undefended | SN-Tune | RSN-Tune | SN-Orig | RSN-Orig |
|--------------|------------|---------|----------|---------|----------|
| StrongREJECT | 0.149      | 0.177   | 0.139    | 0.110   | 0.126    |
| MMLU-Pro     | 0.195      | 0.205   | 0.200    | 0.193   | 0.207    |

#### Mistral-7B-Instruct-v0.2, default defense, 1 seed

No evidence that SN-Tune and RSN-Tune defend against fine-tuning.

**GSM8K fine-tuning lr=2e-5** (catastrophic forgetting):

|              | Before | Undefended | SN-Tune | RSN-Tune | SN-Orig | RSN-Orig |
|--------------|--------|------------|---------|----------|---------|----------|
| StrongREJECT | 0.548  | 0.092      | 0.146   | 0.137    | 0.140   | 0.154    |
| MMLU-Pro     | 0.343  | 0.118      | 0.123   | 0.148    | 0.118   | 0.125    |

**GSM8K fine-tuning lr=5e-6**:

|              | Before | Undefended | SN-Tune | RSN-Tune | SN-Orig | RSN-Orig |
|--------------|--------|------------|---------|----------|---------|----------|
| StrongREJECT | 0.548  | 0.289      | 0.326   | 0.322    | 0.304   | 0.317    |
| MMLU-Pro     | 0.343  | 0.293      | 0.302   | 0.305    | 0.296   | 0.286    |

#### Running the actual original codebase

We also ran the [original Safety-Neuron
codebase](https://github.com/zhaoyiran924/Safety-Neuron) on Llama-2-7B-Chat and
Mistral-7B-Instruct-v0.2k, rather than using the `match_original_code` flag in
our own implementation, to rule out implementation differences. However,
parameter comparison revealed that the saved defended checkpoints were identical
to the base models. Our interpretation of what the paper says the training
hyperparameters are lead to very little training: 50 samples (section 3.1)
effective batch size of 32 (`train_neuron.py` in the codebase), and learning
rate of 1e-6 (section 3.1) leads to only 2 steps at a very low learning rate,
leading to little change in the model.

### Conclusion

We were unable to reproduce the paper's Table 4 findings. We did not find
evidence that RSN-Tune preserved safety better than SN-Tune or the undefended
baseline after GSM8K fine-tuning.

- **No RSN-Tune < SN-Tune < Undefended ordering.** The paper reports RSN-Tune
  reducing harmful score from 41.0 (Undefended) to 26.0, with SN-Tune at 38.0.
  In our experiments, all post-fine-tuning conditions are similar.
- **Implementation doesn't matter.** We tested our paper-based formulation,
  our reimplementation of the original codebase's algorithm, and the original
  codebase itself. Neither produces the expected result.
- We think the defense needs to be launched with a `num_training_samples`
  greater than 50.

Possible explanations for the discrepancy with the paper:
1. The paper uses average ASR across four adversarial attacks (Direct, GCG,
   AutoDAN, PAIR) via HarmBench, which may be more sensitive to defense effects
   than our StrongREJECT direct-prompting metric.
2. The paper does not specify GSM8K fine-tuning hyperparameters. Our fine-tuning
   may not exactly match theirs. We followed the setup of Qi et al. 2024.
3. The effect may be real but smaller than our measurement noise (~0.02 std on
   StrongREJECT).
4. We're reporting MMLU-Pro score, whereas the paper is reporting accuracy on
   the downstream task GSM8K, in which case catastrophic forgetting might not
   show up or be relevant.

### Paper reference (Table 4)

|         | Before | Undefended | SN-Tune | RSN-Tune |
|---------|-------:|-----------:|--------:|---------:|
| GSM8K   |   16.8 |     26.5   |    27.2 |     26.2 |
| Harmful |    0.0 |     41.0   |    38.0 |     26.0 |

(Harmful = average ASR across Direct Attack, GCG, AutoDAN, and PAIR on AdvBench.)
