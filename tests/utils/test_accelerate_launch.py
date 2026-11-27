"""Tests for `run_with_accelerate` (runs on CPU with a single process)."""

from pathlib import Path

import pytest

from tamperbench.whitebox.utils.ops import fsdp_launch_args, run_with_accelerate

CPU_LAUNCH_ARGS = ["--cpu", "--num_machines", "1", "--num_processes", "1", "--mixed_precision", "no"]


def write_marker(path: str, text: str) -> None:
    """Write `text` to `path` (module-level so the launched process can import it)."""
    Path(path).write_text(text)


def fail() -> None:
    """Raise an error inside the launched process."""
    raise ValueError("boom")


def test_runs_target_with_args(tmp_path: Path) -> None:
    """The target runs in the launched process with the given arguments."""
    marker = tmp_path / "marker.txt"
    run_with_accelerate(write_marker, (str(marker), "hello"), launch_args=CPU_LAUNCH_ARGS)
    assert marker.read_text() == "hello"


def test_failure_raises() -> None:
    """A failing target surfaces as a RuntimeError in the caller."""
    with pytest.raises(RuntimeError, match="my context"):
        run_with_accelerate(fail, launch_args=CPU_LAUNCH_ARGS, error_context="my context")


def test_rejects_non_importable_target() -> None:
    """Lambdas and nested functions can't be imported by name in the launched process."""
    with pytest.raises(ValueError, match="module-level"):
        run_with_accelerate(lambda: None, launch_args=CPU_LAUNCH_ARGS)


def test_fsdp_launch_args_sets_num_processes() -> None:
    """The FSDP launch args enable FSDP with the requested number of processes."""
    launch_args = fsdp_launch_args(num_processes=4)
    assert "--use_fsdp" in launch_args
    assert launch_args[launch_args.index("--num_processes") + 1] == "4"
