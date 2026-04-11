ARG PYTORCH_CUDA_VERSION=2.9.0-cuda12.8-cudnn9
FROM pytorch/pytorch:${PYTORCH_CUDA_VERSION}-runtime
LABEL org.opencontainers.image.source=https://github.com/criticalml-uw/TamperBench

ENV DEBIAN_FRONTEND=noninteractive

# Install some useful packages
RUN apt-get update -q \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    # essential for running
    # gcc is needed for vLLM's backend
    gcc git git-lfs tini \
    # nice to have for devbox development
    curl vim tmux less sudo rsync wget \
    # CircleCI
    ssh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.0-python3.10-alpine /usr/local/bin/uv /usr/local/bin/uv

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

# Pre-bake dependencies into a venv at /home/dev/.venv. The repo itself is not
# included in the image as we usually want to clone it and checkout `main` or a
# specific commit.
# After cloning, symlink the pre-built venv into the repo so that tools
# (basedpyright, IDEs, uv) find it at the conventional .venv location:
#   ln -s /home/dev/.venv /workspace/TamperBench/.venv
ENV PATH="/home/dev/.venv/bin:/home/dev/.local/bin:$PATH"
# Create venv and install dependencies (but not the project itself).
# The cache mount keeps uv's download cache out of the image but persists it
# across builds. UV_LINK_MODE=copy is needed because hardlinks into the cache
# mount would dangle in the final image.
RUN --mount=type=cache,target=/home/dev/.cache/uv,uid=${USERID},gid=${GROUPID} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    UV_PROJECT_ENVIRONMENT=/home/dev/.venv UV_LINK_MODE=copy \
    uv sync --locked --no-install-project

WORKDIR /workspace

# Default command to run -- may be changed at runtime
CMD ["/bin/bash"]
