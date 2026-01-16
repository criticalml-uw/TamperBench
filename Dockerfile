ARG PYTORCH_CUDA_VERSION=2.9.0-cuda12.8-cudnn9
# Need devel version for flash-attn to compile
FROM pytorch/pytorch:${PYTORCH_CUDA_VERSION}-devel

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
COPY --from=ghcr.io/astral-sh/uv:0.9.22-python3.10-alpine /usr/local/bin/uv /usr/local/bin/uv

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

# Install dependencies into a venv outside the repo directory. The repo will be
# cloned at runtime (see k8s/devbox.yaml), but the venv with deps is pre-baked.
# UV_PROJECT_ENVIRONMENT tells uv where to create the venv instead of .venv.
#
# This env var affects all uv projects in the container, so installing a
# different uv project will clobber the venv. If that becomes a problem, then
# make UV_PROJECT_ENVIRONMENT local to the uv commands in the Dockerfile and
# then create a symlink from /home/dev/.venv into /workspace/SafeTuneBed/.venv
# after cloning in the k8s/ YAMLs.
ENV UV_PROJECT_ENVIRONMENT=/home/dev/.venv
ENV PATH="/home/dev/.venv/bin:/home/dev/.local/bin:$PATH"

# Create venv and install dependencies (but not the project itself).
# flash_attn is memory intensive to compile, MAX_JOBS decreases the memory usage
# though compilation will be slower. If building locally you may need to go into
# "Settings > Resources" in Docker Desktop and bump the memory limit.
# (https://github.com/Dao-AILab/flash-attention#installation-and-features)
# Run Pyright so its Node.js package gets installed.
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    MAX_JOBS=1 uv sync --all-groups --locked --no-install-project \
    && uv run basedpyright --version

WORKDIR /workspace

# Default command to run -- may be changed at runtime
CMD ["/bin/bash"]
