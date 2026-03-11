#!/usr/bin/env python3
"""One-time setup: create the GCS bucket for sweep results."""

from gcp import BUCKET, ZONE, check_not_in_docker, gcloud

check_not_in_docker()

# Extract region from zone (e.g., us-west1-c -> us-west1)
region = ZONE.rsplit("-", 1)[0]

gcloud(
    "storage", "buckets", "create", BUCKET,
    f"--location={region}",
    "--uniform-bucket-level-access",
    check=False,  # Bucket may already exist
)
