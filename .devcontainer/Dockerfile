# ── Stage 1: Builder ─────────────────────────────────────────────────
# Full devel image for compiling native extensions during uv sync.
# This stage is discarded — only /opt/venv and CUDA dev tools are kept.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
WORKDIR /x/workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --extra dev


# ── Stage 1b: Collect QdstrmImporter + deps ─────────────────────────
# ldd can only resolve co-located libs inside the builder where they exist,
# so we collect them here into a staging directory for the final COPY.
FROM builder AS nsys-importer-deps
RUN mkdir -p /opt/nvidia/nsight-systems/importer-deps \
    && SRC=/opt/nvidia/nsight-compute/2025.1.1/host/linux-desktop-glibc_2_11_3-x64 \
    && cp "$SRC/QdstrmImporter" /opt/nvidia/nsight-systems/importer-deps/ \
    && ldd "$SRC/QdstrmImporter" \
       | grep "$SRC" | awk '{print $3}' \
       | xargs -I{} cp -L {} /opt/nvidia/nsight-systems/importer-deps/


# ── Stage 2: Final ───────────────────────────────────────────────────
# Slim runtime base — no static archives, no nsight-compute GUI.
# nvcc, headers, and nsys CLI are copied from the builder.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# System packages
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
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv

# Node.js + Claude Code (for devcontainer use)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# CUDA dev tools from builder (nvcc, headers, nvvm — NO static archives)
# These enable torch.utils.cpp_extension to compile custom CUDA ops.
COPY --from=builder /usr/local/cuda-12.8/bin/           /usr/local/cuda-12.8/bin/
COPY --from=builder /usr/local/cuda-12.8/include/       /usr/local/cuda-12.8/include/
COPY --from=builder /usr/local/cuda-12.8/nvvm/          /usr/local/cuda-12.8/nvvm/
COPY --from=builder /usr/local/cuda-12.8/lib64/stubs/   /usr/local/cuda-12.8/lib64/stubs/

# Nsight Systems CLI + runtime libs from the nsight-compute bundle.
# The full nsight-systems package is ~622 MB; this cherry-picks just the
# profiler CLI (~268 MB) and the QdstrmImporter + deps (~75 MB) needed to
# convert raw traces into .nsys-rep / SQLite.
COPY --from=builder /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/ \
                    /opt/nvidia/nsight-systems/target-linux-x64/
# QdstrmImporter + its shared lib deps (~75 MB). Collected in the builder
# stage where ldd can resolve them, then copied as a unit.
COPY --from=nsys-importer-deps /opt/nvidia/nsight-systems/importer-deps/ \
                               /opt/nvidia/nsight-systems/target-linux-x64/
# Remove the bundled libstdc++ to avoid poisoning the system linker.
# The system's libstdc++ (from Ubuntu 24.04) is newer and compatible.
RUN rm -f /opt/nvidia/nsight-systems/target-linux-x64/libstdc++.so.6
# QdstrmImporter needs its co-located libs at runtime; nsys invokes it as a
# subprocess, so we wrap it to set LD_LIBRARY_PATH only for that process.
RUN mv /opt/nvidia/nsight-systems/target-linux-x64/QdstrmImporter \
       /opt/nvidia/nsight-systems/target-linux-x64/QdstrmImporter.bin \
    && printf '#!/bin/sh\nLD_LIBRARY_PATH=/opt/nvidia/nsight-systems/target-linux-x64 exec /opt/nvidia/nsight-systems/target-linux-x64/QdstrmImporter.bin "$@"\n' \
       > /opt/nvidia/nsight-systems/target-linux-x64/QdstrmImporter \
    && chmod +x /opt/nvidia/nsight-systems/target-linux-x64/QdstrmImporter
RUN ln -s /opt/nvidia/nsight-systems/target-linux-x64/nsys /usr/local/bin/nsys

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
