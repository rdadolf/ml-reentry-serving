#!/usr/bin/env bash
# vm_setup.sh — Runs on the GCP VM (not inside a container).
#
# Clones the repo at a pinned commit and builds the container image.
# Called by cloud_launch.py after VM creation.
#
# Expected environment variables (set by cloud_launch.py):
#   CLONE_URL  — HTTPS clone URL with embedded token
#   BRANCH     — Git branch to check out
#   COMMIT     — Exact commit SHA to pin
#   IMAGE_TAG  — Full Artifact Registry image:tag to pull
set -euo pipefail

echo "=== vm_setup.sh ==="
echo "Branch: $BRANCH"
echo "Commit: $COMMIT"
date

REPO_DIR="$HOME/repo"

# ── Clone repo ────────────────────────────────────────────────────────
echo "--- Cloning repo ---"
git clone --branch "$BRANCH" --single-branch "$CLONE_URL" "$REPO_DIR"
cd "$REPO_DIR"
git checkout "$COMMIT"
echo "Checked out $(git rev-parse --short HEAD) on $BRANCH"

# ── Install Docker ────────────────────────────────────────────────────
echo "--- Installing Docker ---"
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io
sudo usermod -aG docker "$USER"

# ── Pull container image ─────────────────────────────────────────────
echo "--- Pulling container image ---"
echo "Image: $IMAGE_TAG"
gcloud auth configure-docker "$(echo "$IMAGE_TAG" | cut -d/ -f1)" --quiet
sudo docker pull "$IMAGE_TAG"
sudo docker tag "$IMAGE_TAG" sweep:latest

echo "=== Setup complete ==="
date
