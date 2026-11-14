# T-Vaccine Paper Replication (Llama-2-7B)

Replicates Table 1 (Llama-2-7B, p=0.1) from the [T-Vaccine paper](https://arxiv.org/abs/2407.07844).

## Pipeline

```bash
# Phase 1: Alignment with T-Vaccine defense
uv run python scripts/t_vaccine/harden.py --tier llama2
# Phase 2: Fine-tuning attack with poisoned SST2+BeaverTails data
uv run python scripts/t_vaccine/attack.py --tier llama2
# Phase 3: Evaluate harmful score and SST2 accuracy
uv run python scripts/t_vaccine/evaluate.py --tier llama2
```

## Expected results

From the paper (Table 1, Llama-2-7B, p=0.1):

| Metric | Expected |
|--------|----------|
| Harmful Score (HS) | 14.97% |
| Fine-tuning Accuracy (FA) | 92.40% |

These numbers match with the SST2 column in Table 3, indicating that these
numbers came from using SST2 as the fine-tuning task.

## Output directory

All artifacts are written under `data/t_vaccine_hardened/meta-llama_Llama-2-7b-hf/`:

```
alignment/          # Phase 1 LoRA adapter
attack/             # Phase 2 LoRA adapter
data/sst2.json      # Built SST2 benign dataset
eval/               # Phase 3 evaluation outputs
  harmful_predictions.json
  harmful_eval.json
  sst2_eval.json
  summary.json
logs/               # Training logs
```
