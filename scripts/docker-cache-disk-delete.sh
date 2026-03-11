#!/usr/bin/env bash
# Delete the Docker cache persistent disk.
# Only run if you no longer need the cached layers.
set -euo pipefail

gcloud compute disks delete reentry-docker-cache \
  --project=research-489502 \
  --zone=us-west1-b \
  --quiet

echo "Disk deleted: reentry-docker-cache"
