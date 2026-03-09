#!/usr/bin/env bash
# Run a sweep inside the local devcontainer.
# MLflow env vars are already set in the Dockerfile.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
exec python exp/vllm-sweeps/run-sweep.py "$@"
