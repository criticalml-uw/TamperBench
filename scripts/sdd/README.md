# SDD Replication: Vanilla vs. SDD on Llama2-7b-chat

Replicating the main defense result from Table 1 of
["SDD: Self-Degraded Defense against Malicious Fine-tuning"](https://arxiv.org/abs/2507.21182)
(Chen et al., ACL 2025).

## Paper results (Table 1, Llama2-7b-chat)

The paper uses the LLM-finetune-Safety benchmark (Qi et al., 2023) with a
GPT-4 judge scoring harmfulness on a 1-5 scale.

HS = Harmfulness Score (1-5, lower = safer). HR = Harmfulness Rate (% of
score-5 responses).

 | Method  | Initial            | 10-shot MFT         | 50-shot MFT         | 100-shot MFT        |
 | ------- | ------------------ | ------------------- | ------------------- | ------------------- |
 | Vanilla | HS 1.06 / HR 0.3%  | HS 3.58 / HR 50.0%  | HS 4.52 / HR 80.3%  | HS 4.54 / HR 80.0%  |
 | SDD     | HS 2.14 / HR 0%    | HS 2.14 / HR 0%     | HS 2.57 / HR 0%     | HS 1.57 / HR 0%     |


## Our results

Evaluated with GPT-4o judge on HEx-PHI (300 prompts, 11 harm categories).
MFT attack: full-parameter fine-tuning on AdvBench with lr=5e-5, batch size
10, 5 epochs (Qi et al. 2023 settings). SDD defense: lr=2e-5, 500 steps,
batch size 24, 8K samples.

Harmfulness (HS / HR):

 | Method  | Initial            | 10-shot MFT         | 50-shot MFT         | 100-shot MFT        |
 | ------- | ------------------ | ------------------- | ------------------- | ------------------- |
 | Vanilla | HS 1.13 / HR 1.0%  | HS 4.94 / HR 96.3%  | HS 5.00 / HR 99.7%  | HS 4.86 / HR 91.0%  |
 | SDD     | HS 1.00 / HR 0.0%  | HS 1.58 / HR 13.7%  | HS 4.87 / HR 93.0%  | HS 4.84 / HR 89.7%  |

MMLU-Pro:

 | Method  | Initial | 10-shot MFT | 50-shot MFT | 100-shot MFT |
 | ------- | ------- | ----------- | ----------- | ------------ |
 | Vanilla | 23.2%   | 21.4%       | 12.5%       | 13.8%        |
 | SDD     | 20.9%   | 23.4%       | 3.2%        | 1.3%         |

### Key findings

- Strong defense against 10-shot MFT: HR drops from 96% (vanilla) to 14%
  (SDD). The defense resisted the weakest.
- Self-degradation visible but harmfulness is still visible: Under 50/100-shot
  MFT, MMLU-Pro collapses to 1-3% for SDD (vs 12-14% for vanilla), confirming
  the self-degradation mechanism — the model's general capabilities degrade
  under attack. However, the harmfulness is still high, unlike the paper's
  results.

## Pipeline

```
scripts/sdd/
  config.py          # Model tiers, output paths
  harden.py          # Phase 1: Apply SDD defense
  attack.py          # Phase 2: MFT attack + evaluation (--num-harmful 0 for eval-only)
  submit_slurm.sh    # Submit all jobs to SLURM
```

**Phase 1 (harden):** Train model on 8K `<harmful prompt, irrelevant response>`
pairs from BeaverTails + Alpaca/LIMA with cosine similarity filtering.
Training data is in conversational format so trl.SFTTrainer applies the
model's native chat template (e.g. `[INST]...[/INST]` for Llama2-chat).

**Phase 2 (attack + eval):** Full-parameter fine-tune with k harmful AdvBench
samples (k = 10, 50, 100), then evaluate with PolicyEval + StrongREJECT +
MMLU-Pro. Use `--num-harmful 0` to evaluate without attack. Runs for both
SDD-defended and vanilla baselines.

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
python scripts/sdd/attack.py --tier llama2_chat --num-harmful 0              # SDD initial eval
python scripts/sdd/attack.py --tier llama2_chat --num-harmful 0 --no-defense # Vanilla initial eval
```

Set `SDD_REPO_DIR`, `SDD_UV`, and `SDD_PARTITION` environment variables to
configure the SLURM script for your cluster.

## Differences from paper

- **SDD learning rate:** We use lr=2e-5 instead of the paper's 5e-7. At 5e-7
  with `trl.SFTTrainer`, training loss was flat and the defense had no effect.
  The paper uses LLaMA-Factory which may have different optimizer defaults
  (warmup, weight decay, etc.) that make 5e-7 effective.
- **SDD training framework:** Paper uses LLaMA-Factory for SFT; we use
  `trl.SFTTrainer`.
- **MFT attack data:** We use `AlignmentResearch/AdvBench`'s `proxy_gen_target`
  field as harmful responses, which are short stubs. The paper's actual harmful
  training data was never released.
- **MFT attack hyperparameters:** We use Qi et al. (2023) settings
  (full-parameter, lr=5e-5, batch 10, 5 epochs). The SDD paper doesn't
  specify attack hyperparameters in detail. Our attack appears stronger than
  theirs (vanilla HR 91% vs 80%).
- **Eval prompt set:** We use the public version of the HEx-PHI dataset (300
  prompts x 10 categories). The paper's benchmark has 330 prompts (30 × 11
  categories).
- **Harmfulness judge model:** Paper uses GPT-4; we use GPT-4o. Same judge
  prompt template (from Qi et al. 2023).
- **Utility eval:** Paper reports MMLU and OpenBookQA; we report MMLU-Pro
  (harder variant, so scores are lower in absolute terms).
