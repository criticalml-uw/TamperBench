# MMLU-Pro Prompt Format Comparison

Experiment comparing four MMLU-Pro prompt formats across all model-attack pairs to determine the best default prompting strategy.

## Variants tested

[few-shot, zero-shot] x [raw text continuation, chat templating]

For base models without a native chat template, the corresponding instruct model's template was borrowed (e.g. `meta-llama/Llama-3.1-8B-Instruct` for `meta-llama/Meta-Llama-3-8B`).

For each attack I took the `best.json` hyperparameters from
/data/saad_hossain/SafeTuneBed/results/nov7_trial (the StrongREJECT-maximizing
hyperparameters from that set of runs).

I skipped embedding attacks because they hit a bug my run script and decided
they weren't a high enough priority to re-run and wait for the results for.

## Recommendation

**Use few-shot continuation prompting** (`n_shots=5, use_chat_template=False`), the existing default.

Few-shot continuation is the top-performing variant for the large majority of models, both when averaged across all attacks and when looking at unattacked (`no_weight_modification`) accuracy alone. This holds for base models, most instruct models, and defense-hardened models.

### Per-model average accuracy

| Model                 | fewshot_cont | zeroshot_cont | fewshot_chat | zeroshot_chat |
| ---                   | ---          | ---           | ---          | ---           |
| llama3_1b_base        | **0.0911**   | 0.0509        | 0.0857       | 0.0705        |
| llama3_1b_instruct    | **0.1508**   | 0.1151        | 0.1341       | 0.1437        |
| llama3_3b_base        | **0.1968**   | 0.1190        | 0.1738       | 0.1222        |
| llama3_3b_instruct    | **0.3048**   | 0.2738        | 0.2913       | 0.2897        |
| llama3_8b_base        | **0.3151**   | 0.1635        | 0.2540       | 0.1548        |
| llama3_8b_instruct    | 0.3984       | 0.3413        | **0.4357**   | 0.3405        |
| llama3_8b_lat         | **0.3635**   | 0.2738        | 0.3333       | 0.2992        |
| llama3_8b_refat       | **0.3397**   | 0.2579        | 0.3159       | 0.2825        |
| llama3_8b_rr          | 0.3730       | 0.3040        | **0.3738**   | 0.3254        |
| llama3_8b_tar         | 0.1080       | 0.0920        | 0.1107       | **0.1196**    |
| llama3_8b_triplet_adv | **0.3661**   | 0.2938        | 0.3625       | 0.2937        |
| mistral_7b_base       | **0.3238**   | 0.1571        | 0.3063       | 0.0524        |
| mistral_7b_instruct   | 0.2159       | 0.1659        | **0.2540**   | 0.2127        |
| qwen3_0_6b            | **0.1857**   | 0.1452        | 0.1413       | 0.1683        |
| qwen3_0_6b_base       | **0.2270**   | 0.2159        | 0.2198       | 0.1397        |
| qwen3_1_7b            | **0.2952**   | 0.2389        | 0.2468       | 0.2159        |
| qwen3_1_7b_base       | **0.3571**   | 0.3063        | 0.3222       | 0.2079        |
| qwen3_4b              | **0.4867**   | 0.4316        | 0.4000       | 0.3990        |
| qwen3_4b_base         | **0.5159**   | 0.4921        | 0.5008       | 0.4016        |
| qwen3_8b              | 0.4992       | **0.5048**    | 0.4619       | 0.3746        |
| qwen3_8b_base         | **0.5429**   | 0.4848        | 0.5348       | 0.4339        |

### Per-model no_weight_modification (unattacked) accuracy

| Model                 | fewshot_cont | zeroshot_cont | fewshot_chat | zeroshot_chat |
| ---                   | ---          | ---           | ---          | ---           |
| llama3_1b_base        | **0.1500**   | 0.0357        | 0.1286       | 0.0571        |
| llama3_1b_instruct    | 0.1786       | 0.1500        | **0.2429**   | 0.2286        |
| llama3_3b_base        | **0.2857**   | 0.1643        | 0.2071       | 0.1214        |
| llama3_3b_instruct    | 0.3143       | 0.2643        | 0.3429       | **0.3571**    |
| llama3_8b_base        | **0.3714**   | 0.1929        | 0.3000       | 0.1214        |
| llama3_8b_instruct    | 0.4214       | 0.4000        | **0.5071**   | 0.4786        |
| llama3_8b_lat         | **0.4214**   | 0.3500        | 0.4143       | 0.3714        |
| llama3_8b_refat       | **0.4214**   | 0.3786        | 0.4071       | 0.3429        |
| llama3_8b_rr          | **0.4357**   | 0.3571        | 0.4357       | 0.3571        |
| llama3_8b_tar         | 0.0929       | 0.1500        | 0.1714       | **0.2214**    |
| llama3_8b_triplet_adv | **0.4643**   | 0.3714        | 0.4000       | 0.3429        |
| mistral_7b_base       | **0.3429**   | 0.0714        | 0.3357       | 0.1429        |
| mistral_7b_instruct   | 0.2857       | **0.2929**    | 0.2643       | 0.2357        |
| qwen3_0_6b            | 0.2857       | 0.2643        | 0.2286       | **0.2929**    |
| qwen3_0_6b_base       | **0.3000**   | 0.2643        | 0.2500       | 0.1143        |
| qwen3_1_7b            | **0.4286**   | 0.4143        | 0.3929       | 0.3500        |
| qwen3_1_7b_base       | **0.3714**   | 0.3357        | 0.3500       | 0.2500        |
| qwen3_4b              | **0.5786**   | 0.5571        | 0.5286       | 0.5357        |
| qwen3_4b_base         | 0.4929       | **0.5714**    | 0.5214       | 0.4571        |
| qwen3_8b              | 0.5643       | **0.6357**    | 0.5500       | 0.4500        |
| qwen3_8b_base         | **0.6214**   | 0.5929        | 0.5714       | 0.5714        |

## Scripts

- `run_mmlu_pro_variants.py` -- Run all 4 variants for a single model-attack pair.
- `run_one.sh` -- Wrapper to run a single model-attack pair, for testing before queueing all variants
- `submit_all.sh` -- Submit SLURM jobs for all model-attack combinations.
- `analyze_results.py` -- Aggregate results and print per-model accuracy tables.
