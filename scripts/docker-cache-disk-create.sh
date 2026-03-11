#!/usr/bin/env bash
# Create a persistent disk for Docker layer caching.
# Run once. The disk persists across VM lifecycles.
#
# Usage:
#   ./scripts/docker-cache-disk-create.sh <zone>
set -euo pipefail

ZONE="${1:?Usage: $0 <zone>}"

gcloud compute disks create reentry-vllm-docker \
  --project=research-489502 \
  --zone="$ZONE" \
  --size=50GB \
  --type=pd-standard

echo "Disk created: reentry-vllm-docker (50 GB, $ZONE)"
echo "Attach to VMs via --disk=name=reentry-vllm-docker,auto-delete=no"
