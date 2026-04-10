# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportUnusedCallResult=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportPrivateImportUsage=false
# ruff: noqa: F722
"""Utility functions for model operations and Hugging Face Hub interactions."""

import os
import subprocess
from pathlib import Path

import einops
import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError  # type: ignore
from jaxtyping import Float
from torch import Tensor

# Removed logging import - not needed for this module

api = HfApi()


def is_launched_with_accelerate():
    """Check if the current process was launched with Accelerate.

    Returns:
        True if launched with Accelerate, False otherwise
    """
    # Method 1: Check environment variables set by Accelerate
    if os.environ.get("ACCELERATE_MIXED_PRECISION"):
        return True
    return False


def is_huggingface_path(name_or_path: Path) -> bool:
    """Check if identifier refers to a Hugging Face Hub repository.

    Returns True for valid HF repositories (including private/gated).
    Returns False for local paths or non-existent repositories.
    """
    if name_or_path.expanduser().exists():  # local directory / file
        return False

    try:
        api.model_info(str(name_or_path))  # anonymous if token is None
        return True  # public repo
    except HfHubHTTPError as e:
        if e.response.status_code in (401, 403):  # private or gated
            raise RuntimeError(
                f"Invalid Huggingface access token for {name_or_path}.\n"
                "Set the environment variable HF_TOKEN to a valid access token."
            ) from e
        if e.response.status_code == 404:  # no such repo
            raise RuntimeError(f"Model {name_or_path} not found on Hugging Face Hub") from e
        raise  # network error, etc.


def download_model_from_huggingface(name_or_path: Path, download_dir: Path, num_workers: int = 8) -> None:
    """Download a model from the Hugging Face Hub."""
    # download model from Hugging Face Hub
    # Construct the huggingface-cli download command
    download_cmd = [
        "huggingface-cli",
        "download",
        str(name_or_path),
        "--cache-dir",
        str(download_dir),
        "--max-workers",
        str(num_workers),
    ]
    # Execute the download command
    _ = subprocess.run(download_cmd, check=True, capture_output=True, text=True)


def get_orthogonalized_matrix(
    matrix: Float[Tensor, "... d_model"],
    vec: Float[Tensor, "d_model"],
    weight_scale=None,
) -> Float[Tensor, "... d_model"]:
    """Orthogonalize a matrix by removing the component along a given vector.

    Args:
        matrix: The matrix to orthogonalize
        vec: The vector direction to remove from the matrix
        weight_scale: Optional weight scaling factor

    Returns:
        The orthogonalized matrix
    """
    original_dtype = matrix.dtype

    if vec.device != matrix.device:
        vec = vec.to(matrix.device)

    # Handle FP8 quantized weights with weight_scale
    if str(original_dtype) == "torch.float8_e4m3fn" and weight_scale is not None:
        # 1. Convert to FP32 and apply scale
        # if weight_scale.shape[0] == matrix.shape[0]:
        #     pass
        # elif weight_scale.shape[0] == matrix.shape[1]:
        #     weight_scale = weight_scale.T
        # else:
        #     raise ValueError("weight scale sizes don't match the matrix")

        matrix_fp32 = matrix.to(torch.float32) * weight_scale

        # 2. Normalize vector and convert to FP32
        vec = vec.to(torch.float32)
        vec_fp32 = (vec / torch.norm(vec)).to(torch.float32)

        # 3. Perform orthogonalization in FP32
        proj = (
            einops.einsum(matrix_fp32, vec_fp32.unsqueeze(-1), "... d_model, d_model single -> ... single") * vec_fp32
        )
        result_fp32 = matrix_fp32 - proj

        # # 4. Convert back using original scale
        # result = (result_fp32 / weight_scale).to(torch.float8_e4m3fn)
        # Implement above line with clamping to avoid NaN values when translating back to fp8
        max_val = torch.finfo(torch.float8_e4m3fn).max
        min_val = torch.finfo(torch.float8_e4m3fn).min
        clamped_result = torch.clamp(result_fp32 / weight_scale, min_val, max_val)
        return clamped_result.to(torch.float8_e4m3fn)
    else:
        # For non-FP8 weights or when no scale is provided
        matrix_bf16 = matrix.to(torch.bfloat16)
        vec = vec.to(torch.float32)
        vec_bf16 = (vec / torch.norm(vec)).to(torch.bfloat16)

        proj = (
            einops.einsum(matrix_bf16, vec_bf16.unsqueeze(-1), "... d_model, d_model single -> ... single") * vec_bf16
        )

        return (matrix_bf16 - proj).to(original_dtype)
