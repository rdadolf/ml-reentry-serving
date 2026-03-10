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
STATUS_PATH="$BUCKET/sweep-$RUN_ID/status.json"
VM_NAME="$(hostname)"
mkdir -p "$RESULTS_DIR/mlflow"
chmod 777 "$RESULTS_DIR" "$RESULTS_DIR/mlflow"

# ── Status helpers ─────────────────────────────────────────────────────
write_status() {
    local status="$1"
    local error="${2:-}"
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local json="{\"status\":\"$status\",\"run_id\":\"$RUN_ID\",\"vm\":\"$VM_NAME\",\"branch\":\"$BRANCH\",\"commit\":\"$COMMIT\",\"timestamp\":\"$ts\""
    if [ -n "$error" ]; then
        # Escape quotes and newlines for JSON
        error="$(echo "$error" | sed 's/"/\\"/g' | tr '\n' ' ')"
        json="$json,\"error\":\"$error\""
    fi
    json="$json}"
    echo "$json" | gcloud storage cp - "$STATUS_PATH" 2>/dev/null || true
}

# Write "failed" status on any error exit
on_error() {
    local exit_code=$?
    local error_text=""
    # Grab last line of the run log for the status summary
    if [ -f "$HOME/run.log" ]; then
        error_text="$(tail -1 "$HOME/run.log")"
    fi
    write_status "failed" "$error_text"
    # Upload the full run.log to GCS for debugging
    if [ -f "$HOME/run.log" ]; then
        gcloud storage cp "$HOME/run.log" "$BUCKET/sweep-$RUN_ID/run.log" 2>/dev/null || true
    fi
    exit "$exit_code"
}
trap on_error ERR

# ── Mark running ───────────────────────────────────────────────────────
write_status "running"

# ── Run sweep ─────────────────────────────────────────────────────────
GPU_FLAG=""
if nvidia-smi &>/dev/null; then
    GPU_FLAG="--gpus all"
    echo "--- Running sweep (GPU) ---"
else
    echo "--- Running sweep (CPU) ---"
fi
sudo docker run --rm \
    $GPU_FLAG \
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

# ── Upload results ───────────────────────────────────────────────────
echo "--- Uploading results ---"
gcloud storage rsync "$RESULTS_DIR" "$BUCKET/sweep-$RUN_ID/" --recursive
echo "Results uploaded to $BUCKET/sweep-$RUN_ID/"

# ── Mark complete ─────────────────────────────────────────────────────
write_status "complete"
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
