#!/usr/bin/env bash
# Create a persistent disk for Docker layer caching.
# Run once. The disk persists across VM lifecycles.
set -euo pipefail

gcloud compute disks create reentry-vllm-docker \
  --project=research-489502 \
  --zone=us-west1-b \
  --size=50GB \
  --type=pd-standard

echo "Disk created: reentry-vllm-docker (50 GB, us-west1-b)"
echo "Attach to VMs via --disk=name=reentry-vllm-docker,auto-delete=no"
