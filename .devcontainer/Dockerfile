# ── Stage 1: Builder ─────────────────────────────────────────────────
# Full devel image for compiling native extensions during uv sync.
# This stage is discarded — only /opt/venv and CUDA dev tools are kept.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
WORKDIR /x/workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --extra dev


# ── Stage 2: Final ───────────────────────────────────────────────────
# Slim runtime base — no static archives, no nsight-compute.
# nvcc and headers are copied from the builder for custom CUDA op compilation.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# System packages (no nsight-systems — saves ~622 MB)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        clang \
        lldb \
        lld \
        clang-format \
        clang-tidy \
        make \
        cmake \
        git \
        curl \
        ssh \
        vim \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

ENV CC=clang
ENV CXX=clang++

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Node.js + Claude Code (for devcontainer use)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# CUDA dev tools from builder (nvcc, headers, nvvm — NO static archives)
# These enable torch.utils.cpp_extension to compile custom CUDA ops.
COPY --from=builder /usr/local/cuda-12.6/bin/           /usr/local/cuda-12.6/bin/
COPY --from=builder /usr/local/cuda-12.6/include/       /usr/local/cuda-12.6/include/
COPY --from=builder /usr/local/cuda-12.6/nvvm/          /usr/local/cuda-12.6/nvvm/
COPY --from=builder /usr/local/cuda-12.6/lib64/stubs/   /usr/local/cuda-12.6/lib64/stubs/

# Rename the base image's 'ubuntu' user (UID 1000) to 'devel'
RUN usermod -l devel -d /home/devel -m ubuntu \
    && groupmod -n devel ubuntu

# Python deps from builder (pre-built venv)
COPY --chown=devel:devel --from=builder /opt/venv /opt/venv

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"
WORKDIR /x/workspace
COPY pyproject.toml uv.lock ./

USER devel
