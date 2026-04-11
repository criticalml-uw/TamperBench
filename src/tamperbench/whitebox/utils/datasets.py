"""Utilities for downloading and caching datasets."""

import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import datasets
import nltk

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
    url_path = urllib.parse.urlparse(url).path
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
    url_path = urllib.parse.urlparse(url).path
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

    elif data_format == "csv":
        text = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    else:
        raise ValueError(f"Unsupported format: {data_format}")


def get_dataset(source: str | Path, data_format: str | None = None, force_download: bool = False) -> Any:
    """Load a dataset from a URL or local path.

    Args:
        source: URL, local file path, or Path object.
        data_format: File format ('jsonl', 'json', 'txt'). Auto-detected if None.
        force_download: Re-download if cached (URLs only).

    Returns:
        Parsed dataset (list of dicts for JSONL, dict for JSON, str for TXT).
    """
    content = None
    cache_path = None
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        data_format = data_format or guess_format_from_url(source)
        if not data_format:
            raise ValueError(f"Could not guess format from URL: {source}")
        cache_path = get_cache_path(source, data_format)
        if force_download or not cache_path.exists():
            with urllib.request.urlopen(source) as response:
                content = response.read()
            cache_path.write_bytes(content)
    else:
        cache_path = Path(source)
        if not cache_path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        data_format = data_format or cache_path.suffix.lstrip(".")
        if not data_format:
            raise ValueError(f"Could not determine format: {source}")

    if not content:
        content = cache_path.read_bytes()

    return parse_data(content, data_format)


def get_config_cache_path(cache_key: str, config: dict[str, Any], data_format: str = "jsonl") -> Path:
    """Get the cache file path for a dataset based on configuration parameters.

    Args:
        cache_key: Identifier for this dataset (e.g., dataset name, operation type).
        config: Dictionary of parameters that affect the dataset output.
        data_format: File format extension (default: "jsonl").

    Returns:
        Path: The path where the dataset should be cached.
    """
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    safe_key = cache_key.replace("/", "_").replace(":", "_")
    cache_dir = DATA_DIR / safe_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f"{config_hash}.{data_format}"


def save_cached_dataset(dataset: datasets.Dataset, cache_key: str, config: dict[str, Any]) -> Path:
    """Save a dataset to cache as JSONL.

    Args:
        dataset: The dataset to save.
        cache_key: Identifier for this dataset.
        config: Dictionary of parameters that affect the dataset output.

    Returns:
        Path: The path where the dataset was saved.
    """
    cache_path = get_config_cache_path(cache_key, config, data_format="jsonl")
    dataset.to_json(cache_path, orient="records", lines=True)
    return cache_path


def ensure_nltk_resource(resource_path: str, download_name: str | None = None) -> None:
    """Ensure an NLTK resource is available, downloading if necessary.

    Not thread-safe, as one thread may start a download, and then another
    thread may see the file and return from ensure_nltk_resource(), falsely
    considering the download to be done. But at that point the second thread
    cannot safely use the download yet.

    Args:
        resource_path: The NLTK resource path (e.g., "taggers/averaged_perceptron_tagger_eng").
        download_name: The name to pass to nltk.download(). Defaults to last component of resource_path.
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        name = download_name or resource_path.split("/")[-1]
        nltk.download(name, quiet=True)
