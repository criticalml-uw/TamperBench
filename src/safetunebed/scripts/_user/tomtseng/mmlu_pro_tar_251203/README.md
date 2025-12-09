# Summary

This directory is debugging why the official TAR models had poor MMLU-Pro score.

We saw a MMLU-Pro score of 16% for the Llama-3-8B TAR model, vs. 44% for Llama-3-8B-Instruct. Some of the gap is from poor instruction following in the sense that the TAR model sometimes doesn't output the letter answer, but it also seems the TAR model is actually just worse at MMLU-Pro. Indeed, the TAR paper (Tamirisa et al. 2024) also shows in Table 1 that the TAR bio model is worse at MMLU, with its score dropping from 67.3% to 54.7

## Details

- Reformatting the questions doesn't help: I tried zero-shot prompting as well as toggling the chat template but the TAR model still was consistently worse.
- Regex parser isn’t missing things: I parsed the responses from our existing `nov7_trial` experiments with an LLM-as-a-judge and got a similar score of 14%.
- There is poor instruction following, though that's not the whole gap: in the existing experiments, the TAR model fails to output a letter answer 22% of the time, vs. Llama-3-8B-Instruct’s 0% of the time. If I parse answers with an LLM-as-a-judge that is also given the full context (question, multiple choice options, and ground-truth-answer), and ask it whether the TAR model gave a response clearly matching the ground-truth answer, the TAR model gets a higher score of 24%.

## Other remarks

It varies by model which formatting works best:
- The TAR model and Llama-3.1-8B-Instruct are better at zero-shot, whereas Llama-3-8B-Instruct is better at few-shot.
- Within few-shot, Llama-3-8B-Instruct is better without the chat template (44% without chat template vs 11% with), whereas Llama-3.1-8B-Instruct is better with (41% without vs 49% with)
- (The original MMLU-Pro codebase does few-shot with no chat template.)
