#!/usr/bin/env bash
# cloud-entrypoint.sh — Runs inside the container on a cloud VM.
#
# Sets up the environment (uv sync) and runs the sweep. This is the
# cloud equivalent of what devcontainer.json's postCreateCommand +
# manual invocation do locally.
set -euo pipefail

cd /x/workspace
uv sync --extra dev --quiet
exec python exp/vllm-sweeps/run-sweep.py "$@"
