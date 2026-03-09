#!/usr/bin/env bash
# run_on_vm.sh — Runs on the GCP VM (not inside a container).
#
# Executes a sweep inside the pre-built container, uploads results to
# GCS, and optionally stops or deletes the VM.
#
# Expected environment variables (set by cloud_run.py):
#   HF_TOKEN     — Hugging Face token for model downloads
#   BUCKET       — GCS bucket (gs://...)
#   RUN_ID       — Identifier for this sweep run (MMDD-HHMM)
#   BRANCH       — Git branch (for metadata)
#   COMMIT       — Commit SHA (for metadata)
#   AFTER_RUN    — "none", "stop", or "delete"
#   PROJECT      — GCP project ID (needed for self-delete)
#   VM_ZONE      — GCP zone (needed for self-delete)
#
# Any additional arguments are passed through to run-sweep.py.
set -euo pipefail

echo "=== run_on_vm.sh ==="
echo "Run ID:    $RUN_ID"
echo "After run: $AFTER_RUN"
date

REPO_DIR="$HOME/repo"
RESULTS_DIR="$HOME/results/$RUN_ID"
mkdir -p "$RESULTS_DIR/mlflow"

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
    sweep:latest \
    bash /x/workspace/.devcontainer/cloud-entrypoint.sh "$@"

# ── Upload results ────────────────────────────────────────────────────
echo "--- Uploading results ---"
gcloud storage rsync "$RESULTS_DIR" "$BUCKET/sweep-$RUN_ID/" --recursive
echo "Results uploaded to $BUCKET/sweep-$RUN_ID/"

# ── Post-run cleanup ─────────────────────────────────────────────────
echo "=== Sweep complete ==="
date

case "$AFTER_RUN" in
    stop)
        echo "Stopping VM in 60s..."
        sleep 60
        sudo shutdown -h now
        ;;
    delete)
        echo "Deleting VM in 60s..."
        sleep 60
        gcloud compute instances delete "$(hostname)" \
            --zone="$VM_ZONE" --project="$PROJECT" --quiet
        ;;
    *)
        echo "VM left running."
        ;;
esac
