FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# System packages: Python 3.12, git, nsight-systems
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        git \
        nsight-systems \
        curl \
        ssh \
        vim \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Node.js (for Claude Code) via NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# Rename the base image's 'ubuntu' user (UID 1000) to 'devel'
RUN usermod -l devel -d /home/devel -m ubuntu \
    && groupmod -n devel ubuntu

# Install deps to /opt/venv so the bind mount doesn't clobber them.
# UV_PROJECT_ENVIRONMENT persists, so postCreateCommand's `uv sync` is a no-op.
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
WORKDIR /x/workspace
COPY pyproject.toml ./
RUN uv sync --extra dev \
    && chown -R devel:devel /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

# MLflow: SQLite backend on the persistent bind mount
ENV MLFLOW_TRACKING_URI="sqlite:////home/devel/mlflow/mlflow.db"
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT="/home/devel/mlflow/artifacts"

# Alias for mlflow ui (doesn't respect MLFLOW_TRACKING_URI)
RUN echo 'alias mlflow-ui="mlflow ui --backend-store-uri \$MLFLOW_TRACKING_URI"' >> /home/devel/.bashrc

USER devel
