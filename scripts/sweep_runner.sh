#!/usr/bin/env bash
# sweep_runner.sh — Runs on the GCP VM (not inside a container).
#
# This script is copied to the VM by launch_cloud.py and executed via
# nohup so it survives SSH disconnects. It clones the repo, builds the
# container, runs the sweep, uploads results, and optionally terminates.
#
# Expected environment variables (set by launch_cloud.py):
#   CLONE_URL  — HTTPS clone URL with embedded token
#   BRANCH     — Git branch to check out
#   COMMIT     — Exact commit SHA to pin
#   HF_TOKEN   — Hugging Face token for model downloads
#   BUCKET     — GCS bucket (gs://...)
#   RUN_ID     — Identifier for this sweep run
#   KEEP_ALIVE — "1" to skip auto-terminate, "0" to shut down after upload
set -euo pipefail

echo "=== sweep_runner.sh ==="
echo "Branch: $BRANCH"
echo "Commit: $COMMIT"
echo "Run ID: $RUN_ID"
echo "Keep alive: $KEEP_ALIVE"
date

REPO_DIR="$HOME/repo"
RESULTS_DIR="$HOME/results"
mkdir -p "$RESULTS_DIR/mlflow"

# ── Clone repo ────────────────────────────────────────────────────────
echo "--- Cloning repo ---"
git clone --branch "$BRANCH" --single-branch "$CLONE_URL" "$REPO_DIR"
cd "$REPO_DIR"
git checkout "$COMMIT"
echo "Checked out $(git rev-parse --short HEAD) on $BRANCH"

# ── Build container ───────────────────────────────────────────────────
echo "--- Building container ---"
IMAGE_TAG="sweep:$RUN_ID"
docker build -f .devcontainer/Dockerfile -t "$IMAGE_TAG" .

# ── Run sweep ─────────────────────────────────────────────────────────
echo "--- Running sweep ---"
docker run --rm \
    --gpus all \
    -v "$REPO_DIR:/x/workspace" \
    -v "$RESULTS_DIR:/results" \
    -e "HF_TOKEN=$HF_TOKEN" \
    -e "MLFLOW_TRACKING_URI=sqlite:////results/mlflow/mlflow.db" \
    -e "MLFLOW_DEFAULT_ARTIFACT_ROOT=/results/mlflow/artifacts" \
    -e "RUN_ID=$RUN_ID" \
    -e "BRANCH=$BRANCH" \
    -e "COMMIT=$COMMIT" \
    "$IMAGE_TAG" \
    bash /x/workspace/.devcontainer/cloud-entrypoint.sh

# ── Upload results ────────────────────────────────────────────────────
echo "--- Uploading results ---"
gcloud storage rsync "$RESULTS_DIR" "$BUCKET/sweep-$RUN_ID/" --recursive
echo "Results uploaded to $BUCKET/sweep-$RUN_ID/"

# ── Cleanup ───────────────────────────────────────────────────────────
echo "=== Sweep complete ==="
date

if [ "$KEEP_ALIVE" = "0" ]; then
    echo "Auto-terminating VM in 60s..."
    sleep 60
    sudo shutdown -h now
fi
