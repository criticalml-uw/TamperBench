# Defense cumulative-max curves

Context: Our [current defense
results](https://farinc.slack.com/archives/C095L3PLBGF/p1772367795699299)
suggest that defenses provide no advantage against our attack hyperparameter
sweeps. The top harmfulness score achieved against each defended model is about
the same as the score against the corresponding undefended model.

Question: Do the defenses at least increase the number of trials needed to find
a good attack, even if they don't raise the ceiling on attack success?

Method: For each (model, attack) pair, we plot the cumulative maximum
StrongREJECT score across Optuna hyperparameter-search trials (x-axis = trial
number, y-axis = best StrongREJECT score so far allowing a 10pp MMLU-Pro drop).

**Result**: The defenses do not meaningfully increase sample complexity. Defended
curves generally rise just as fast as the undefended curve.
The exception is CTRL, which sometimes shows a lower ceiling, but CTRL also
heavily damages MMLU.

## Usage

1. Extract trial data from the sweep results (run on `cais`):
   ```
   python3 extract_sweep_data.py
   ```

2. Generate plots:
   ```
   python3 plot_cummax_sweep.py
   ```

   Outputs PNGs to `plots/`.
