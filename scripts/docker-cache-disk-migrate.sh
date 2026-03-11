#!/usr/bin/env bash
# Migrate the Docker cache persistent disk to a different zone.
#
# Usage:
#   ./scripts/docker-cache-disk-migrate.sh <source-zone> <target-zone>
#
# Example:
#   ./scripts/docker-cache-disk-migrate.sh us-west1-b us-west1-a
set -euo pipefail

PROJECT="research-489502"
DISK="reentry-vllm-docker"
SNAPSHOT="${DISK}-migration"

SOURCE_ZONE="${1:?Usage: $0 <source-zone> <target-zone>}"
TARGET_ZONE="${2:?Usage: $0 <source-zone> <target-zone>}"

if [ "$TARGET_ZONE" = "$SOURCE_ZONE" ]; then
  echo "Target zone is the same as source zone ($SOURCE_ZONE). Nothing to do."
  exit 0
fi

echo "Migrating disk '$DISK' from $SOURCE_ZONE to $TARGET_ZONE..."

# 1. Snapshot the existing disk
echo "Creating snapshot '$SNAPSHOT'..."
gcloud compute disks snapshot "$DISK" \
  --project="$PROJECT" \
  --zone="$SOURCE_ZONE" \
  --snapshot-names="$SNAPSHOT"

# 2. Create new disk from the snapshot in the target zone
echo "Creating disk '$DISK' in $TARGET_ZONE from snapshot..."
gcloud compute disks create "$DISK" \
  --project="$PROJECT" \
  --zone="$TARGET_ZONE" \
  --source-snapshot="$SNAPSHOT" \
  --type=pd-standard

# 3. Delete the old disk
echo "Deleting old disk in $SOURCE_ZONE..."
gcloud compute disks delete "$DISK" \
  --project="$PROJECT" \
  --zone="$SOURCE_ZONE" \
  --quiet

# 4. Clean up the snapshot
echo "Deleting migration snapshot..."
gcloud compute snapshots delete "$SNAPSHOT" \
  --project="$PROJECT" \
  --quiet

echo "Done. Disk '$DISK' is now in $TARGET_ZONE."
