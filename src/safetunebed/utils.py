"""General utilities for safetunebed."""

from pathlib import Path

import git


def get_repo_root() -> Path:
    """Get the root directory of the git repository."""
    repo = git.Repo(".", search_parent_directories=True)
    working_dir = repo.working_tree_dir
    if working_dir is None:
        raise RuntimeError("Could not find git repository root")
    return Path(working_dir)
