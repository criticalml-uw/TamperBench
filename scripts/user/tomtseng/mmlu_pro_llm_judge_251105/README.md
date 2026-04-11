Note: The contents of this directory used a lot of LLM assistance and the code has not
been reviewed carefully.

# MMLU-Pro LLM-as-a-judge evaluation

Checks the MMLU-Pro responses from Oct 29 2025 runs (runs for our first arXiv
release): do the extracted regex results match up with LLM-as-a-judge results?

If there is a large mismatch, that could indicate that the regex results fail to
capture some knowledge capabilities, and are instead partially measuring gap
that comes from reduced ability to carefully instruction-follow and answer the
question in the expected format.

Files:
- `extract_sample_output.py` samples a bunch of MMLU-Pro responses and the
  regex-parsed single-letter answers from the responses
- `judge.py`: LLM judge for extracting MMLU-Pro single-letter answers from
  responses
- `run_judge.py`: Runs judge on each sampled response and checks that the LLM
  judge answers match the regex-parsed answers, printing mismatches to
  /tmp/judge_mismatches.txt.

## Results Analysis

Ran the LLM judge (GPT-4) on 6,468 MMLU-Pro responses and compared with regex-based answer extraction. Overall agreement: **97.8%** (144 mismatches).

Overall there is good agreement between the regex extraction and the
LLM-as-a-judge.

### LLM judge found no answer, regex did (98 cases, 68% of mismatches)

The most common type of mismatch. Usually here the regex is wrong: the regex fallback pattern looks for the last standalone capital letter (`\b[A-J]\b`), which often is not an actual attempt at answering the question.

**Common subcategories:**

a) **Pathological repetitive text**
   - Same sentence repeated 50-200+ times (Rows 14, 40, 56, 84, 120, 156, 182, 238, 420, 732), no answer given

b) **Truncated/incomplete responses** (Rows 42, 326):
   - Response cuts off mid-sentence: `A subset H of a group (G,*) is a group if`
   - No actual answer in text, but regex finds letter

c) **Complex formatted text** (Rows 438, 483, 663):
   - Truth tables with many rows of symbolic logic
   - Regex picks up a capital letter "F" from table structure

### LLM judge is wrong

The judge can get confused:
- Judge returns an answer that is not in the response (e.g., rows 219, 369, 719)
- Response contains multiple answers, it's ambiguous what answer the judge
  should pick (e.g., rows 102, 434)

### Regex misses legitimate answers

Non-standard but clear phrasings the regex doesn't catch:
- Row 2168: "Le choix correct est le b."
- Row 270: "The purchase of a new car by a car rental company would be classified under C."
- Row 419: "we can conclude that (H) is the correct answer"
- Row 748: "The answer 'Tdc' appears in (E)" → Regex: C, Judge: E ✓
- Row 912: "therefore, (E) must be the correct answer"
- Row 3599: "answer (D)"
 - Note that regex parsing for `(<answer>)` may also give false results in
   the opposite direction. Row 102: "The answer (B) is not correct."

### Recommendations for Future Work

1. Try ablating the final fallback regex:
```python
def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
```
This is the regex responsible for most of the cases where the regex incorrectly
finds an answer when the LLM judge does not. What happens if we remove this
regex? It'll remove a lot of those wrong cases---but conversely our analysis
here does not consider that it may also remove correct cases where this final
fallback regex catches the correct answer.

2. There are a lot of responses where the response continues by generating new
   questions and answering them. This can be confusing to parse (both from a
   regex and an LLM-as-a-judge standpoint). This is probably a result of the
   few-shot format, unclear MMLU-Pro instructions (which is the fault of the
   MMLU-Pro authors, not us), and of models being incapable.

   Consider stripping anything after "\nQuestion:" from the responses to cut
   most cases of these confabulated questions. (Hopefully there are very few
   cases where models actually legitimately output "\nQuestion:" as a rhetorical
   device in their CoT reasoning.)

3. If re-running this LLM-as-a-judge, modify the system prompt to make the judge
   respond "n/a" if the response spams a bunch of inconsistent answers.

### Limitations

A reasonable conclusion is that since only 2% of answers were mismatched, the
regex evaluation is good enough. However it's possible that mismatched answers
were concentrated in some models/attacks. As an extreme case, imagine that all
the mismatched answers came from a single attacked model. In that case we cannot
trust the MMLU-Pro regex scores for that model.
