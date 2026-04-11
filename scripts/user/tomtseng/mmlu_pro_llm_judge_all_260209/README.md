# MMLU-Pro LLM judge (all files)

Runs gpt-4.1-mini as a judge-with-context on all nov7_trial parquet files,
comparing LLM-judge accuracy against regex-based answer extraction.

## Conclusion

Using an LLM judge is not clearly worth the cost for MMLU-Pro evaluation:

Results with gpt-4.1-mini as a judge:
- Out of 7311 parquet files in nov7_trial, only 95 show a >=5 p.p. increase
  in MMLU-Pro score from using the judge-with-context, and 61 of those 95 are
  on llama3_8b_tar. The judge helps most when models give correct answers in
  prose (e.g., in French after multilingual finetuning) without outputting a
  letter choice.
- 1755 files show a >=5 p.p. *decrease* from using the judge.
  - Most seem to be runaway responses (model gives correct answer then generates
    extra questions/answers, or otherwise provide a bunch of nonsense text after
    the answer).
  - Sometimes the judge marks an answer as wrong because the reasoning leading
    up to the model giving the letter answer is wrong or says the letter answer
    is wrong.
- The judge can also be fooled to mark answers as correct by degenerate responses like
  "assistantassistantassistant..." repeated. Upgrading to gpt-4.1 fixes this.
- Using gpt-4.1-mini on all files cost ~$150, and this would grow if we
  increased the MMLU-Pro sample size (currently 140 as of 2026 Feb 9). Using
  gpt-4.1 would 5x the cost.

## 2026-02-18: GPT-5 mini judge

Re-ran with low-reasoning gpt-5-mini as the judge model. GPT-5 mini is
substantially better than gpt-4.1-mini:
- No longer fooled by degenerate responses like "assistantassistant...".
- Tighter overall distribution: max diffs compared to regex parser shrink from
  [-0.16, +0.47] to [-0.09, +0.19].
- Only 1.6% rows have the judge disagreeing with the regex about whether the
  answer is correct. The judge is more accurate than the regex on the disagreements:
  * when regex says it's correct and judge says it's incorrect, it's usually:
    * the regex picks up a spurious letter that's accidentally right,
    * the response contains multiple answers, the regex grabs the first one which happens to be right but judge correctly marks it as wrong
    * the justification the response gives leading up to its answer suggests a different answer than what the model finally outputs
  * when judge says it's correct and regex says it's incorrect, it's usually that the model explained things in prose correctly but didn't output the answer in the expected format

So the judge is an improvement over regex. However, it doesn't change our
results much (`analyze_judge_impact.py`):
- Baseline `no_weight_modification` MMLU-Pro scores change only by 0--2pp,
  except for TAR which gains 7pp under the judge
- For the hyperparameter sweeps where we're looking for the top StrongREJECT
  under a max 10 pp MMLU-Pro drop, 178/190=94% model-attack pairs have no
  change. 12 pairs change, with a mean change of 0.003. The two biggest changes:
  - qwen3_32b_prev / full_parameter_finetune: +0.12 (judge admits 1 more trial above threshold that happens to have high harmfulness)
  - qwen3_4b / full_parameter_finetune: +0.12 (same pattern — 4 more trials pass the judge threshold)
