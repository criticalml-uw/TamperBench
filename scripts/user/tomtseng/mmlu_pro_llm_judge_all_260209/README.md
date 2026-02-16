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
