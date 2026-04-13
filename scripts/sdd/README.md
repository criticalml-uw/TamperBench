# SDD Replication: Vanilla vs. SDD on Llama2-7b-chat

Replicating the main defense result from Table 1 of
["SDD: Self-Degraded Defense against Malicious Fine-tuning"](https://arxiv.org/abs/2507.21182).

## Target results (Table 1, Llama2-7b-chat)

The paper uses the LLM-finetune-Safety benchmark (Qi et al., 2023) with a
GPT-4 judge scoring harmfulness on a 1-5 scale.

| Method  | Initial | 10-shot MFT | 50-shot MFT | 100-shot MFT |
|---------|---------|-------------|-------------|--------------|
| Vanilla | HS 1.06 / HR 0.3% | HS 3.58 / HR 50.0% | HS 4.52 / HR 80.3% | HS 4.54 / HR 80.0% |
| SDD     | HS 2.14 / HR 0%    | HS 2.14 / HR 0%     | HS 2.57 / HR 0%     | HS 1.57 / HR 0%     |

HS = Harmfulness Score (1-5, lower = safer). HR = Harmfulness Rate (% of score-5 responses).

General capability (Table 2):

| Method  | MMLU  | OpenBookQA |
|---------|-------|------------|
| Vanilla | 46.35 | 33.40      |
| SDD     | 47.04 | 33.00      |

## Pipeline

```
scripts/sdd/
  config.py          # Model tiers, output paths
  harden.py          # Phase 1: Apply SDD defense
  attack.py          # Phase 2: MFT attack + evaluation
  submit_slurm.sh    # Submit all jobs to SLURM
```

**Phase 1 (harden):** Train model on 8K `<harmful prompt, irrelevant response>`
pairs from BeaverTails + Alpaca/LIMA. Paper defaults: lr=5e-7, 500 steps,
batch size 24.

**Phase 2 (attack + eval):** LoRA fine-tune with k harmful AdvBench samples
(k = 10, 50, 100), then evaluate. Runs for both SDD-defended and vanilla
baselines.

## Usage

```bash
# Dry-run to see SLURM scripts
bash scripts/sdd/submit_slurm.sh --dry-run

# Submit to cluster
bash scripts/sdd/submit_slurm.sh

# Or run phases individually
python scripts/sdd/harden.py --tier llama2_chat
python scripts/sdd/attack.py --tier llama2_chat --num-harmful 100
python scripts/sdd/attack.py --tier llama2_chat --num-harmful 100 --no-defense
```

## Differences from paper

### Differences

- **SDD training framework:** Paper uses LLaMA-Factory for SFT; we use
  trl.SFTTrainer. Both do standard SFT but may differ in tokenization details
  and data collation.
- **Harmfulness judge model:** Paper uses GPT-4; PolicyEval defaults to GPT-4o.
  The judge prompt template is identical.
- **Utility eval:** Paper reports MMLU and OpenBookQA; we report MMLU-Pro
  (harder variant, scores will be lower but trends should match).
- **StrongREJECT:** We also run StrongREJECT as an additional safety metric
  not in the original paper.
