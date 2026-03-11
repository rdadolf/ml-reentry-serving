#!/usr/bin/env bash
# Delete the Docker cache persistent disk.
# Only run if you no longer need the cached layers.
#
# Usage:
#   ./scripts/docker-cache-disk-delete.sh <zone>
set -euo pipefail

ZONE="${1:?Usage: $0 <zone>}"

gcloud compute disks delete reentry-vllm-docker \
  --project=research-489502 \
  --zone="$ZONE" \
  --quiet

echo "Disk deleted: reentry-vllm-docker ($ZONE)"
