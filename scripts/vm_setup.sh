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

# ── Build container ───────────────────────────────────────────────────
echo "--- Building container ---"
sudo docker build -f .devcontainer/Dockerfile -t sweep:latest .

echo "=== Setup complete ==="
date
