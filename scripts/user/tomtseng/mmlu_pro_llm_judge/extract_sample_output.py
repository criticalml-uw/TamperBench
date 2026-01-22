"""Samples one random row from each mmlu_pro_val/inferences.parquet file.

This is meant to be run on the CAIS cluster.

Fetches a sample (one row from each Optuna trial from each attacked model) of
MMLU-Pro responses from /data/saad_hossain/SafeTuneBed/results/oct29_trial (runs
for our first arXiv release) and outputs them into /tmp/out.parquet, with
columns:
- file: Full path to the parquet file
- row_index: The randomly sampled row index (0-indexed) from the file
- response: The response text with "PRED: <letter>" removed
- prediction: The extracted letter (A-J) from "PRED: <letter>", or "n/a" if no prediction was found

This is for getting a sampling of what MMLU-Pro outputs look like among all of our models and attacks.
"""

import random
import re
from pathlib import Path

import pandas as pd

# Set seed for reproducibility (optional - remove if you want different results each time)
random.seed(42)

# Find all matching files
parquet_files = list(
    Path("/data/saad_hossain/SafeTuneBed/results/oct29_trial").rglob("mmlu_pro_val/inferences.parquet")
)

data = []

for file_path in sorted(parquet_files):
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)

        # Get a random row from the response column
        if "response" in df.columns and len(df) > 0:
            # Sample a random index
            random_idx = random.randint(0, len(df) - 1)
            sampled_response = df["response"].iloc[random_idx]

            # Extract prediction from "PRED: <letter>" at the end
            # Look for "PRED:" followed by optional whitespace and optional letter
            pred_pattern = r"PRED:\s*([A-Z])?"
            match = re.search(pred_pattern, sampled_response)

            if match:
                prediction = match.group(1) if match.group(1) else "n/a"
                # Remove the "PRED: <answer>" part from response
                response_cleaned = re.sub(pred_pattern + r".*$", "", sampled_response).strip()
            else:
                prediction = "n/a"
                response_cleaned = sampled_response

            data.append(
                {
                    "file": str(file_path),
                    "row_index": random_idx,
                    "response": response_cleaned,
                    "prediction": prediction,
                }
            )
        else:
            data.append(
                {
                    "file": str(file_path),
                    "row_index": -1,
                    "response": "[No response column or empty file]",
                    "prediction": "n/a",
                }
            )
    except Exception as e:
        data.append(
            {
                "file": str(file_path),
                "row_index": -1,
                "response": f"[Error reading file: {e}]",
                "prediction": "n/a",
            }
        )

# Create DataFrame
result_df = pd.DataFrame(data)

# Save as parquet (efficient and preserves types)
output_path = Path("/tmp/out.parquet")
result_df.to_parquet(output_path, index=False)

print(f"Extracted {len(result_df)} entries and saved to /tmp/out.parquet")
print(f"\nDataFrame shape: {result_df.shape}")
print(f"\nColumn names: {list(result_df.columns)}")
print("\nFirst few rows:")
print(result_df.head())
print("\nRow index statistics:")
print(result_df["row_index"].describe())
print("\nPrediction value counts:")
print(result_df["prediction"].value_counts())
