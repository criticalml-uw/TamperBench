"""Data types for CTRL curation."""

import hashlib
from dataclasses import dataclass

from vllm import SamplingParams


@dataclass
class Input:
    """A single input (query, response pair)."""

    query: str
    response: str

    def to_key(self) -> str:
        """Generate unique key for caching."""
        return hashlib.md5(f"{self.query}:{self.response}".encode()).hexdigest()


@dataclass
class Prompt:
    """A query with a prompt and sampling parameters."""

    query: str
    prompt: str
    sampling_params: SamplingParams


@dataclass
class Variant:
    """A generated variant with metrics."""

    query: str  # Which original input this came from (constant across rounds)
    text: str
    temp: float
    top_p: float
    perplexity: float | None = None
    readability: float | None = None
    helpfulness: float | None = None
    source_hash: str | None = None  # Hash of the source response that was curated

    def has_metrics(self) -> bool:
        """Check if all metrics are computed."""
        return self.perplexity is not None and self.readability is not None and self.helpfulness is not None

    @staticmethod
    def hash_source(text: str) -> str:
        """Hash source text for cache key."""
        return hashlib.md5(text.encode()).hexdigest()[:8]
