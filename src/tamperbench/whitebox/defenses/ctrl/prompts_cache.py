"""Cache management for CTRL defense prompt optimization."""

import fcntl
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from tamperbench.whitebox.defenses.ctrl.types import Input, Variant


class Cache:
    """Manages caching of inputs, variants, and results across rounds."""

    def __init__(self, cache_dir: Path):
        """Initialize cache and load existing data.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir: Path = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self.inputs: dict[str, Input] = {}
        self.variants: dict[str, list[Variant]] = {}  # query -> list of variants
        self.completed_round: int = -1  # Last fully completed round (-1 = none)
        self.current_best: list[Variant] = []  # Best variants from last completed round

        # Track which queries this worker has modified (for safe concurrent saves)
        self._modified_queries: set[str] = set()

        self._load()

    @property
    def cache_file(self) -> Path:
        """Get the path to the cache file."""
        return self.cache_dir / "cache.json"

    def _load(self):
        """Load all cached data."""
        if not self.cache_file.exists():
            return

        with open(self.cache_file) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return

        # Load inputs
        for key, inp_dict in data.get("inputs", {}).items():
            self.inputs[key] = Input(**inp_dict)

        # Load variants
        for query, var_list in data.get("variants", {}).items():
            self.variants[query] = [
                Variant(
                    query=v.get("query", query),  # Old cache might have input_key
                    text=v["text"],
                    temp=v["temp"],
                    top_p=v["top_p"],
                    perplexity=v.get("perplexity"),
                    readability=v.get("readability"),
                    helpfulness=v.get("helpfulness"),
                    source_hash=v.get("source_hash"),
                )
                for v in var_list
            ]

        # Load round state
        self.completed_round = data.get("completed_round", -1)
        self.current_best = [Variant(**v) for v in data.get("current_best", [])]

    def save(self):
        """Save all cached data.

        Only updates queries that this worker has modified, preserving other workers'
        changes to avoid race conditions in parallel execution.

        Uses file locking and atomic writes to prevent data loss when multiple
        workers save simultaneously.
        """
        lock_file = self.cache_dir / ".cache.lock"

        # Acquire exclusive lock for the entire read-modify-write operation
        with open(lock_file, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                # Load current state from disk to merge with our changes
                current = Cache.__new__(Cache)
                current.cache_dir = self.cache_dir
                current.inputs = {}
                current.variants = {}
                current.completed_round = -1
                current.current_best = []
                current._modified_queries = set()
                current._load()

                # Merge inputs: only update keys we modified
                merged_inputs = dict(current.inputs)
                for k, v in self.inputs.items():
                    merged_inputs[k] = v

                # Merge variants: only update queries we modified, preserving others
                merged_variants = {k: [asdict(v) for v in vs] for k, vs in current.variants.items()}
                for query in self._modified_queries:
                    if query in self.variants:
                        merged_variants[query] = [asdict(v) for v in self.variants[query]]

                data = {
                    "inputs": {k: asdict(v) for k, v in merged_inputs.items()},
                    "variants": merged_variants,
                    "completed_round": max(current.completed_round, self.completed_round),
                    "current_best": [asdict(v) for v in self.current_best]
                    if self.completed_round >= current.completed_round
                    else [asdict(v) for v in current.current_best],
                }

                # Write to temp file first, then atomically rename
                fd, temp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".json.tmp")
                try:
                    with open(fd, "w") as f:
                        json.dump(data, f, indent=2)
                    # Atomic rename
                    Path(temp_path).rename(self.cache_file)
                except Exception:
                    Path(temp_path).unlink(missing_ok=True)
                    raise
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def save_round_state(self, round_idx: int, best_variants: list[Variant]):
        """Save the completed round state for resumption.

        Args:
            round_idx: The round index that was just completed (0-indexed).
            best_variants: The best variants selected after this round.
        """
        self.completed_round = round_idx
        self.current_best = best_variants
        self.save()

    def get_variants(self, query: str) -> list[Variant]:
        """Get all cached variants for given query."""
        return self.variants.get(query, [])

    def get_variant(self, query: str, temp: float, top_p: float, source_hash: str | None = None) -> Variant | None:
        """Get a cached variant for given query, temp, top_p, and source.

        Args:
            query: The query string.
            temp: Temperature used for generation.
            top_p: Top-p value used for generation.
            source_hash: Hash of the source response that was curated.
                         Required for correct cache lookup.
        """
        for v in self.variants.get(query, []):
            if v.temp == temp and v.top_p == top_p and v.source_hash == source_hash:
                return v
        return None

    def save_variants(self, variants: list[Variant]):
        """Save variants to cache, updating metrics if new variant has them."""
        for variant in variants:
            if variant.query not in self.variants:
                self.variants[variant.query] = []

            # Track that we're modifying this query
            self._modified_queries.add(variant.query)

            # Find existing variant with same text + params + source
            existing_idx = None
            for i, v in enumerate(self.variants[variant.query]):
                if (
                    v.text == variant.text
                    and v.temp == variant.temp
                    and v.top_p == variant.top_p
                    and v.source_hash == variant.source_hash
                ):
                    existing_idx = i
                    break

            if existing_idx is not None:
                # Update existing if new one has metrics and old one doesn't
                existing = self.variants[variant.query][existing_idx]
                if variant.has_metrics() and not existing.has_metrics():
                    self.variants[variant.query][existing_idx] = variant
            else:
                # Add new variant
                self.variants[variant.query].append(variant)

        self.save()
