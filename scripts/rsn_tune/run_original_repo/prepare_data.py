"""Download and format datasets for the Safety-Neuron pipeline.

Datasets:
  1. AlignmentResearch/AdvBench       -> harmful_behaviors.txt  (safety neuron detection)
  2. abhayesian/circuit-breakers-dataset -> circuit_breakers_train.json  (SN-tune training)
  3. wikimedia/wikipedia (en)          -> wikipedia_en.txt       (foundation neuron detection)
"""

import argparse
import json
import os


def prepare_harmful_corpus(output_path):
    """Download AdvBench and write harmful prompts to a text file (one per line)."""
    from datasets import load_dataset

    ds = load_dataset("AlignmentResearch/AdvBench", split="train")
    count = 0
    with open(output_path, "w") as f:
        for row in ds:
            # The 'content' column contains the harmful text as a list of strings;
            # 'instructions' is empty in this dataset.
            content = row.get("content", [])
            if isinstance(content, list) and content:
                text = content[0].strip()
            elif isinstance(content, str):
                text = content.strip()
            else:
                text = ""
            if text:
                f.write(text + "\n")
                count += 1
    print(f"Wrote {count} harmful prompts to {output_path}")


def prepare_training_data(output_path):
    """Download circuit-breakers dataset and format as JSON-lines for HF load_dataset('json')."""
    from datasets import load_dataset

    ds = load_dataset("abhayesian/circuit-breakers-dataset", split="train")
    with open(output_path, "w") as f:
        for row in ds:
            record = {"original_question": row["prompt"], "response": row["chosen"]}
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(ds)} training samples to {output_path}")


def prepare_wikipedia_corpus(output_path, num_samples=10000):
    """Download Wikipedia and write first-paragraph excerpts (one per line)."""
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    count = 0
    with open(output_path, "w") as f:
        for row in ds:
            text = row["text"].strip().split("\n")[0].strip()
            if len(text) > 50:
                f.write(text + "\n")
                count += 1
                if count >= num_samples:
                    break
    print(f"Wrote {count} Wikipedia excerpts to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and format datasets for Safety-Neuron.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--wiki-samples", type=int, default=10000)
    parser.add_argument(
        "--only",
        choices=["harmful", "training", "wikipedia"],
        nargs="*",
        help="Only prepare specific datasets (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    targets = set(args.only) if args.only else {"harmful", "training", "wikipedia"}

    if "harmful" in targets:
        prepare_harmful_corpus(os.path.join(args.data_dir, "harmful_behaviors.txt"))
    if "training" in targets:
        prepare_training_data(os.path.join(args.data_dir, "circuit_breakers_train.json"))
    if "wikipedia" in targets:
        prepare_wikipedia_corpus(os.path.join(args.data_dir, "wikipedia_en.txt"), args.wiki_samples)
