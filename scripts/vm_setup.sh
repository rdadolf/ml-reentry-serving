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
git checkout --quiet "$COMMIT"
echo "Checked out $(git rev-parse --short HEAD) on $BRANCH"

# ── Install Docker ────────────────────────────────────────────────────
echo "--- Installing Docker ---"
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io
sudo usermod -aG docker "$USER"

# ── Mount Docker cache disk ──────────────────────────────────────────
CACHE_DEV="/dev/disk/by-id/google-docker-cache"
CACHE_MNT="/mnt/docker-cache"

if [ -b "$CACHE_DEV" ]; then
    echo "--- Mounting Docker cache disk ---"
    sudo mkdir -p "$CACHE_MNT"

    # Format only if no filesystem exists (first use)
    if ! sudo blkid "$CACHE_DEV" &>/dev/null; then
        echo "First use — formatting disk"
        sudo mkfs.ext4 -m 0 -F "$CACHE_DEV"
    fi

    sudo mount "$CACHE_DEV" "$CACHE_MNT"

    # Point Docker's storage at the persistent disk
    sudo mkdir -p /etc/docker
    echo '{"data-root": "/mnt/docker-cache/docker"}' | sudo tee /etc/docker/daemon.json
    sudo systemctl restart docker
    echo "Docker data-root: /mnt/docker-cache/docker"
else
    echo "--- No cache disk attached, using default Docker storage ---"
fi

# ── Pull container image ─────────────────────────────────────────────
echo "--- Pulling container image ---"
echo "Image: $IMAGE_TAG"
sudo gcloud auth configure-docker "$(echo "$IMAGE_TAG" | cut -d/ -f1)" --quiet
sudo docker pull "$IMAGE_TAG"
sudo docker tag "$IMAGE_TAG" sweep:latest

echo "=== Setup complete ==="
date
