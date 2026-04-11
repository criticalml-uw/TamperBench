"""Utilities for running functions in isolated subprocesses."""

# pyright: reportExplicitAny=false, reportAny=false

import multiprocessing
import queue
import traceback
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def _subprocess_wrapper(
    target: Callable[..., T],
    q: "multiprocessing.Queue[dict[str, Any]]",
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Wrap target function execution and put result in queue.

    This wrapper handles exception capture and ensures results are always
    communicated back to the parent process via the queue.

    Args:
        target: The function to execute.
        q: Queue to put results into.
        args: Positional arguments for target.
        kwargs: Keyword arguments for target.
    """
    try:
        result = target(*args, **kwargs)
        q.put({"status": "ok", "payload": result})
    except Exception as error:
        q.put(
            {
                "status": "error",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )


def run_in_isolation(
    target: Callable[..., T],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    error_context: str = "Subprocess",
) -> T:
    """Run a function in an isolated subprocess with proper cleanup.

    This utility handles the common pattern of:
    1. Creating a subprocess with a queue for communication
    2. Running a target function in that subprocess
    3. Properly cleaning up process and queue resources
    4. Raising informative errors if the subprocess fails

    The target function should return a value (not put it in a queue).
    This wrapper handles all the queue communication internally.

    Args:
        target: The function to run in isolation. Should return its result directly.
        args: Positional arguments to pass to the target function.
        kwargs: Keyword arguments to pass to the target function.
        error_context: Context string for error messages (e.g., "StrongReject inference").

    Returns:
        The return value from the target function.

    Raises:
        RuntimeError: If the subprocess fails, with the original error and traceback.

    Example:
        ```python
        def heavy_computation(model_path: str, data: list[str]) -> DataFrame:
            # Load model, run inference, return results
            model = load_model(model_path)
            return model.predict(data)

        # Run in isolated subprocess
        result = run_in_isolation(
            target=heavy_computation,
            args=(model_path, data),
            error_context="Model inference",
        )
        ```
    """
    if kwargs is None:
        kwargs = {}

    q: multiprocessing.Queue[dict[str, Any]] = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_subprocess_wrapper,
        args=(target, q, args, kwargs),
    )
    p.start()

    try:
        # q.get() without timeout blocks in C code and prevents KeyboardInterrupt
        # from being raised. Using a timeout lets Python handle interrupts.
        while True:
            try:
                child_message: dict[str, Any] = q.get(timeout=1.0)
                break
            except queue.Empty:
                if not p.is_alive():
                    raise RuntimeError(f"{error_context}: subprocess died unexpectedly") from None
    finally:
        p.terminate()  # SIGTERM
        p.join(timeout=5)
        if p.is_alive():
            p.kill()  # SIGKILL if SIGTERM didn't work
            p.join(timeout=1)
        p.close()
        q.close()
        q.join_thread()

    if child_message.get("status") != "ok":
        error_msg: str = child_message.get("error", "Unknown error")
        stack_trace: str = child_message.get("traceback", "")
        raise RuntimeError(f"{error_context} failed in subprocess: {error_msg}\n{stack_trace}")

    payload: T = child_message["payload"]
    return payload
