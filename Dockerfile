ARG PYTORCH_CUDA_VERSION=2.0.1-cuda11.7-cudnn8
FROM pytorch/pytorch:${PYTORCH_CUDA_VERSION}-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install some useful packages
RUN apt-get update -q \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    # essential for running
    git git-lfs tini \
    # nice to have for devbox development
    curl vim tmux less sudo rsync wget \
    # CircleCI
    ssh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.17-python3.10-alpine /usr/local/bin/uv /usr/local/bin/uv

# Don't run as root, instead run as non-root user that can still use
# passwordless sudo for admin privileges
ARG USERID=1001
ARG GROUPID=1001
ARG USERNAME=dev
RUN addgroup --gid ${GROUPID} ${USERNAME} \
    && adduser --uid ${USERID} --gid ${GROUPID} --disabled-password --gecos '' ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p "/workspace" \
    && chown -R ${USERNAME}:${USERNAME} "/workspace"
USER ${USERNAME}

# uv will install into VIRTUAL_ENV. We add VIRTUAL_ENV to the path so that users
# of image can use `python` rather than `uv run python`.
ENV VIRTUAL_ENV=/workspace/TamperBench/.venv
ENV PATH="$VIRTUAL_ENV/bin:/home/dev/.local/bin:$PATH"

# Copy package installation files
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml uv.lock /workspace/TamperBench/

# Create venv and install dependencies
# Run Pyright so its Node.js package gets installed
RUN cd /workspace/TamperBench \
    && mkdir -p src/tamperbench \
    && touch src/tamperbench/__init__.py \
    && uv sync --extra dev --frozen \
    && rm -rf "${HOME}/.cache" \
    && basedpyright . \
    && cd .. \
    && rm -rf TamperBench

# Default command to run -- may be changed at runtime
CMD ["/bin/bash"]
