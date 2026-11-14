# Vendored from LiveBench commit 18b524d
# Source: livebench/code_runner/eval/__init__.py
# Modifications:
#   - Rewritten imports to use _vendor relative paths
#   - Use explicit 'fork' multiprocessing context (TamperBench sets 'spawn' as default)
#   - Only includes untrusted_check and its dependencies (not trusted_check, evaluate_files, etc.)
#
# The MIT License
#
# Copyright (c) OpenAI (https://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import contextlib
import io
import multiprocessing
import os
import sys
import time
import types
import unittest
from typing import Dict, Tuple

import numpy as np

from tamperbench.whitebox.evals.livebench_coding._vendor.code_runner_utils import (
    create_tempdir,
    reliability_guard,
    redirect_stdin,
    WriteOnlyStringIO,
    swallow_subprocess_output,
    time_limit,
    safe_environment,
)

# Use explicit fork context since TamperBench sets spawn as default
_fork_ctx = multiprocessing.get_context("fork")

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


def unsafe_execute(
    entry_point: str,
    code: str,
    test_code: str,
    timeout: float,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    stat,  # Value
    details,  # Array
):
    with safe_environment(), create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil
        import builtins

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        # Disable functionalities that can make destructive changes to the test.
        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)
        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        # Set necessary attributes for the module
        new_module.__dict__.update({
            '__builtins__': builtins,
            '__file__': f"{module_name}.py",
            '__package__': None,
            '__doc__': None,
            'sys': sys,
            'os': os,
            'environ': os.environ,
        })

        # Create string IO objects to capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        stdin_capture = WriteOnlyStringIO()

        try:
            full_code = code + "\n" + test_code

            # Use contextlib to redirect stdout and stderr instead of swallowing IO
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture), redirect_stdin(new_target=stdin_capture), swallow_subprocess_output():
                exec(compile(full_code, f"{module_name}.py", 'exec'), new_module.__dict__)
                sys.modules[module_name] = new_module
                TestCases = getattr(new_module, 'TestCases')
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = unittest.TestResult()
                start_time = time.time()
                with time_limit(timeout):
                    suite.run(test_result)

            # Capture stdout and stderr content
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()

            issues = test_result.failures + test_result.errors
            for test, trace in issues:
                details[test.id().split(".")[-1]] = (test.shortDescription(), trace)

            if issues:
                # Store outputs in details
                details["_captured_stdout_"] = stdout_content
                details["_captured_stderr_"] = stderr_content

            stat.value = _SUCCESS
        except BaseException as e:
            # Capture stdout and stderr content before the exception
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()

            # Store outputs and exception details
            details["_captured_stdout_"] = stdout_content
            details["_captured_stderr_"] = stderr_content
            details["_exception_"] = str(e)

            stat.value = _FAILED

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def untrusted_check(
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    min_time_limit: float = 10,
    gt_time_limit: float = 60
) -> Tuple[str, dict]:
    min_time_limit = max(min_time_limit, gt_time_limit)
    timeout = min_time_limit + 1
    # shared memory objects
    stat = _fork_ctx.Value("i", _UNKNOWN)
    manager = _fork_ctx.Manager()
    details = manager.dict()

    p = _fork_ctx.Process(
        target=unsafe_execute,
        args=(
            entry_point,
            code,
            test_code,
            timeout,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            stat,
            details,
        ),
    )
    p.start()
    p.join(timeout=timeout+1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    # convert details to a dict
    details = dict(details)

    if not stat:
        stat = TIMEOUT

    if stat == PASS:
        if details:
            stat = FAIL

    return stat, details
