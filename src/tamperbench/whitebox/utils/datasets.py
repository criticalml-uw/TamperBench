"""Utilities for downloading and caching datasets."""

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from tamperbench.utils import get_repo_root

# Data directory at project root
DATA_DIR = get_repo_root() / "data"


def get_cache_path(url: str, data_format: str) -> Path:
    """Get the cache file path for a given URL.

    Args:
        url: The URL of the dataset.
        data_format: Optional file format extension.

    Returns:
        Path: The path where the dataset should be cached.
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    url_path = urllib.request.urlparse(url).path
    filename = Path(url_path).name

    if not filename.endswith(f".{data_format}"):
        filename = f"{Path(filename).stem}.{data_format}"

    dataset_name = Path(filename).stem
    cache_dir = DATA_DIR / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f"{url_hash}_{filename}"


def guess_format_from_url(url: str) -> str | None:
    """Guess the file format from the URL extension.

    Args:
        url: The URL of the dataset.

    Returns:
        str | None: The guessed format, or None if not recognized.
    """
    url_path = urllib.request.urlparse(url).path
    suffix = Path(url_path).suffix.lstrip(".")
    return suffix or None


def parse_data(content: bytes, data_format: str) -> Any:
    """Parse the downloaded content based on the format.

    Args:
        content: The raw bytes content.
        data_format: The file format (e.g., 'jsonl', 'json', 'csv').

    Returns:
        Any: Parsed data structure.

    Raises:
        ValueError: If the format is not supported.
    """
    if data_format == "jsonl":
        lines = content.decode("utf-8").strip().split("\n")
        return [json.loads(line) for line in lines if line.strip()]

    elif data_format == "json":
        return json.loads(content.decode("utf-8"))

    elif data_format == "txt":
        return content.decode("utf-8")

    else:
        raise ValueError(f"Unsupported format: {data_format}")


def get_dataset(url: str, data_format: str | None = None, force_download: bool = False) -> Any:
    """Download and cache a dataset, or load from cache if available.

    This function downloads a dataset from a URL and caches it locally in the
    data/ directory. On subsequent calls with the same URL, it loads from the
    local cache instead of re-downloading.

    Args:
        url: The URL of the dataset to download.
        data_format: The file format (e.g., 'jsonl', 'json', 'txt'). If None,
            attempts to guess from the URL file extension.
        force_download: If True, re-download even if cached locally.

    Returns:
        Any: The parsed dataset. For JSONL, returns a list of dicts. For JSON,
            returns the parsed JSON structure. For TXT, returns a string.

    Raises:
        ValueError: If the format is not supported.

    Example:
        >>> # Download and parse MT-Bench questions
        >>> questions = get_dataset(
        ...     "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
        ...     "fastchat/llm_judge/data/mt_bench/question.jsonl"
        ... )
        >>> print(f"Loaded {len(questions)} questions")
    """
    data_format = data_format or guess_format_from_url(url)
    if not data_format:
        raise ValueError(f"Could not guess dataset format from URL: {url}")

    cache_path = get_cache_path(url, data_format)

    # Download if needed
    if force_download or not cache_path.exists():
        with urllib.request.urlopen(url) as response:
            content = response.read()

        # Save to cache
        with open(cache_path, "wb") as f:
            f.write(content)
    else:
        # Load from cache
        with open(cache_path, "rb") as f:
            content = f.read()

    # Parse and return
    return parse_data(content, data_format)
