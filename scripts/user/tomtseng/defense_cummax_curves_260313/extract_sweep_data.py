"""Extract trial-level StrongREJECT and MMLU scores from sweep results.

Run on the remote machine:
  python3 extract_sweep_data.py
Writes sweep_data.csv to the same directory as this script.
"""

import csv
import json
import sqlite3
from pathlib import Path

from tqdm import tqdm

BASE = Path("/data/saad_hossain/SafeTuneBed/results/sweeps/seed_42")

# Map directory names to (model_family, defense)
# We infer this from naming conventions
MODEL_FAMILIES = {
    "llama3_8b": "Llama-3-8B",
    "llama3_8b_instruct": "Llama-3-8B-Instruct",
    "qwen3_8b": "Qwen3-8B",
}

DEFENSE_NAMES = {
    "baseline": "Undefended",
    "booster": "Booster",
    "crl": "CRL",
    "ctrl": "CTRL",
    "lat": "LAT",
    "refat": "ReFAT",
    "rr": "R2D2",
    "rsn_tune": "RSN-Tune",
    "tar": "TAR",
    "tar_v2": "TAR-v2",
    "triplet_adv": "Triplet-Adv",
}

SKIP_ATTACKS = {
    "embedding_attack",
    "benign_full_parameter_finetune",
    "benign_lora_finetune",
}

OUT_PATH = Path(__file__).parent / "sweep_data.csv"

with OUT_PATH.open("w", newline="") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(["model_family", "defense", "attack", "trial_number", "strong_reject", "mmlu_pro"])

    for dirpath in tqdm(sorted(BASE.iterdir()), desc="Processing dirs"):
        if not dirpath.is_dir():
            continue
        dirname = dirpath.name

        # Parse model_family and defense from dirname
        model_family = None
        defense_key = None
        for mkey in sorted(MODEL_FAMILIES.keys(), key=len, reverse=True):
            if dirname.startswith(mkey + "_") or dirname == mkey:
                suffix = dirname[len(mkey) :]
                if suffix == "":
                    defense_key = "baseline"
                elif suffix.startswith("_"):
                    defense_key = suffix[1:]
                else:
                    continue
                model_family = MODEL_FAMILIES[mkey]
                break

        if model_family is None:
            continue
        if defense_key not in DEFENSE_NAMES:
            continue

        defense = DEFENSE_NAMES[defense_key]

        for attack_dir in sorted(dirpath.iterdir()):
            if not attack_dir.is_dir() or attack_dir.name in SKIP_ATTACKS:
                continue
            db_path = attack_dir / "optuna_single" / "study.db"
            if not db_path.exists():
                continue

            # Get trial numbers and StrongREJECT scores from study.db.
            # objective=0 is the first (and only) objective, which is StrongREJECT
            # in our single-objective Optuna studies.
            conn = sqlite3.connect(db_path)
            try:
                trials_data = conn.execute(
                    "SELECT t.number, tv.value "
                    "FROM trials t JOIN trial_values tv ON t.trial_id = tv.trial_id "
                    "WHERE t.state = 'COMPLETE' AND tv.objective = 0 "
                    "ORDER BY t.number"
                ).fetchall()
            finally:
                conn.close()

            # Get MMLU scores from evaluation.json files
            attack = attack_dir.name
            for trial_num, sr_score in trials_data:
                mmlu_path = (
                    attack_dir
                    / "optuna_single"
                    / f"trial_{trial_num}"
                    / "tamperbench_evaluation"
                    / "mmlu_pro_val"
                    / "evaluation.json"
                )
                mmlu_score = ""
                if mmlu_path.exists():
                    data = json.loads(mmlu_path.read_text())
                    mmlu_score = data[0]["metric_value"]

                writer.writerow([model_family, defense, attack, trial_num, sr_score, mmlu_score])

print(f"Wrote {OUT_PATH}")
