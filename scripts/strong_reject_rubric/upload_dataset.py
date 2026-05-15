#!/usr/bin/env python3
"""Upload CSV dataset to HuggingFace Hub.

Usage:
    # Upload as private
    uv run scripts/strong_reject_rubric/upload_dataset.py \
        --csv strongreject_targets.csv \
        --repo-id username/strongreject_with_targets \
        --private

    # Upload as public
    uv run scripts/strong_reject_rubric/upload_dataset.py \
        --csv strongreject_targets.csv \
        --repo-id username/strongreject_with_targets

Note: To enable gated access, upload as private first, then go to the
dataset settings on HuggingFace to enable gating.
"""

import argparse
import logging
import os
from pathlib import Path

import datasets
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def upload_csv_to_hub(
    csv_path: Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> None:
    """Upload CSV dataset to HuggingFace Hub.

    Args:
        csv_path: Path to CSV file
        repo_id: Repository ID (e.g., "username/dataset-name")
        token: HuggingFace API token (uses HF_TOKEN env var if not provided)
        private: Whether to make the dataset private
    """
    LOGGER.info(f"Loading CSV from {csv_path}...")

    # Load CSV as HuggingFace Dataset
    dataset = datasets.load_dataset("csv", data_files=str(csv_path), split="train")

    LOGGER.info(f"Dataset loaded with {len(dataset)} rows")
    LOGGER.info(f"Columns: {dataset.column_names}")

    # Upload to HuggingFace Hub
    LOGGER.info(f"Uploading to HuggingFace Hub: {repo_id}")
    LOGGER.info(f"Private: {private}")

    try:
        dataset.push_to_hub(repo_id=repo_id, token=token, private=private)
        LOGGER.info(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")

        if private:
            LOGGER.info("Dataset is private - only you can access it")
            LOGGER.info("To enable gated access: Go to https://huggingface.co/datasets/{repo_id}/settings")

    except Exception as e:
        LOGGER.error(f"Failed to upload to HuggingFace: {e}")
        raise


def main() -> None:
    """Run the main script."""
    parser = argparse.ArgumentParser(
        description="Upload CSV dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to CSV file",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., username/dataset-name)",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (uses HF_TOKEN env var if not provided)",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Make dataset private",
    )

    args = parser.parse_args()

    # Validate CSV exists
    if not args.csv.exists():
        parser.error(f"CSV file does not exist: {args.csv}")

    # Get HF token
    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        parser.error("HuggingFace token not found. Set HF_TOKEN env var or use --hf-token")

    # Upload dataset
    upload_csv_to_hub(csv_path=args.csv, repo_id=args.repo_id, token=token, private=args.private)

    LOGGER.info("Complete!")


if __name__ == "__main__":
    main()
