"""Generic worker for parallelizing GPU work across multiple GPUs.

Each worker will be assigned to a different GPU.
"""

import importlib
import os
import pickle
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import torch

T = TypeVar("T")
R = TypeVar("R")


def worker(
    gpu_id: int,
    inputs_file: Path,
    output_file: Path,
    function_module: str,
    function_name: str,
    kwargs_file: Path,
) -> None:
    """Run a function on a specific GPU in a subprocess.

    Runs in a subprocess with CUDA_VISIBLE_DEVICES set to a specific GPU.

    Args:
        gpu_id: GPU ID to use
        inputs_file: Path to pickled list of inputs
        output_file: Path to save pickled results
        function_module: Module path of function to call (e.g., "my.module")
        function_name: Name of function in module
        kwargs_file: Path to pickled dict of keyword arguments
    """
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Load inputs and kwargs
    print(f"[GPU {gpu_id}] Loading inputs from {inputs_file}")
    with open(inputs_file, "rb") as f:
        inputs: list[Any] = pickle.load(f)

    print(f"[GPU {gpu_id}] Loading kwargs from {kwargs_file}")
    with open(kwargs_file, "rb") as f:
        kwargs: dict[str, Any] = pickle.load(f)

    print(f"[GPU {gpu_id}] Processing {len(inputs)} inputs with {function_module}.{function_name}")

    # Import and call function
    module = importlib.import_module(function_module)
    func: Callable[..., Any] = getattr(module, function_name)

    results = func(inputs, **kwargs)

    # Save results
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"[GPU {gpu_id}] Complete")


def _split_into_batches(items: list[T], num_batches: int) -> list[list[T]]:
    """Split items into roughly equal batches."""
    batch_size = (len(items) + num_batches - 1) // num_batches
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _save_batches_to_files(
    batches: list[list[T]],
    function_kwargs: dict[str, Any],
    temp_dir: Path,
) -> tuple[list[Path], list[Path], Path]:
    """Save batches and kwargs to pickle files for IPC.

    Args:
        batches: List of batches to save
        function_kwargs: Keyword arguments to save
        temp_dir: Directory for temporary files

    Returns:
        Tuple of (input_files, output_files, kwargs_file)
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    kwargs_file = temp_dir / "kwargs.pkl"
    with open(kwargs_file, "wb") as f:
        pickle.dump(function_kwargs, f)

    input_files = []
    output_files = []
    for i, batch in enumerate(batches):
        input_file = temp_dir / f"inputs_gpu_{i}.pkl"
        output_file = temp_dir / f"outputs_gpu_{i}.pkl"
        with open(input_file, "wb") as f:
            pickle.dump(batch, f)
        input_files.append(input_file)
        output_files.append(output_file)

    return input_files, output_files, kwargs_file


def _spawn_worker_processes(
    input_files: list[Path],
    output_files: list[Path],
    kwargs_file: Path,
    function_module: str,
    function_name: str,
) -> list[tuple[int, subprocess.Popen[bytes]]]:
    """Spawn worker processes for each GPU.

    Args:
        input_files: Paths to input pickle files
        output_files: Paths where outputs should be saved
        kwargs_file: Path to kwargs pickle file
        function_module: Module containing the function
        function_name: Name of the function to call

    Returns:
        List of (gpu_id, process) tuples
    """
    worker_script = Path(__file__)
    processes = []

    for gpu_id, (input_file, output_file) in enumerate(zip(input_files, output_files, strict=False)):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable,
            str(worker_script),
            str(gpu_id),
            str(input_file),
            str(output_file),
            function_module,
            function_name,
            str(kwargs_file),
        ]

        print(f"   Launching worker for GPU {gpu_id}")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((gpu_id, proc))

    return processes


def _wait_for_workers(processes: list[tuple[int, subprocess.Popen[bytes]]]) -> None:
    """Wait for all worker processes to complete and check for errors.

    Args:
        processes: List of (gpu_id, process) tuples

    Raises:
        RuntimeError: If any worker process fails
    """
    for gpu_id, proc in processes:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(f"   ERROR: GPU {gpu_id} worker failed with code {proc.returncode}")
            print(stderr.decode() if stderr else "No stderr")
            print(stdout.decode() if stdout else "No stdout")
        else:
            print(f"   GPU {gpu_id} worker completed successfully")

    all_successful = all(proc.returncode == 0 for _, proc in processes)
    if not all_successful:
        raise RuntimeError("One or more GPU workers failed")


def _collect_results(output_files: list[Path]) -> list[R]:
    """Collect and combine results from all worker output files.

    Args:
        output_files: Paths to output pickle files

    Returns:
        Combined results from all workers
    """
    all_results = []
    for output_file in output_files:
        with open(output_file, "rb") as f:
            batch_results: list[R] = pickle.load(f)
            all_results.extend(batch_results)

    print(f"   Collected {len(all_results)} results from all workers")
    return all_results


def parallelize_across_gpus(
    items: list[T],
    function: Callable[..., R],
    function_kwargs: dict[str, Any] | None = None,
    temp_dir: Path | None = None,
) -> list[R]:
    """Parallelize a function across multiple GPUs.

    Splits items into batches (one per GPU) and runs the function on each batch
    in a separate subprocess with its own GPU.

    Args:
        items: List of items to process
        function: Function to call. Must accept (items: list[T], **kwargs) -> list[R]
        function_kwargs: Additional keyword arguments to pass to function
        temp_dir: Directory for temporary IPC files (default: ./parallel_temp)

    Returns:
        Combined results from all GPU workers
    """
    function_kwargs = function_kwargs or {}
    num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        print("   Single GPU/CPU mode - running directly")
        return function(items, **function_kwargs)  # pyright: ignore[reportReturnType]

    print(f"   Multi-GPU mode - parallelizing across {num_gpus} GPUs")

    # Split into batches
    batches = _split_into_batches(items, num_gpus)
    print(f"   Split {len(items)} items into {len(batches)} batches of ~{len(batches[0])} each")

    # Prepare temp directory and files
    if temp_dir is None:
        temp_dir = Path("./parallel_temp")

    input_files, output_files, kwargs_file = _save_batches_to_files(batches, function_kwargs, temp_dir)

    # Get function reference
    function_module = function.__module__
    function_name = function.__name__

    # Spawn workers
    processes = _spawn_worker_processes(input_files, output_files, kwargs_file, function_module, function_name)

    # Wait for completion
    _wait_for_workers(processes)

    # Collect results
    results = _collect_results(output_files)

    for f in input_files + output_files + [kwargs_file]:
        f.unlink(missing_ok=True)
    return results


def main():
    """Entry point for running as a script."""
    if len(sys.argv) != 7:
        print("Usage: worker.py <gpu_id> <inputs_file> <output_file> <function_module> <function_name> <kwargs_file>")
        sys.exit(1)

    gpu_id = int(sys.argv[1])
    inputs_file = Path(sys.argv[2])
    output_file = Path(sys.argv[3])
    function_module = sys.argv[4]
    function_name = sys.argv[5]
    kwargs_file = Path(sys.argv[6])

    worker(
        gpu_id=gpu_id,
        inputs_file=inputs_file,
        output_file=output_file,
        function_module=function_module,
        function_name=function_name,
        kwargs_file=kwargs_file,
    )


if __name__ == "__main__":
    main()
