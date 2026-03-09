#!/usr/bin/env python3
"""Show the state of all cloud resources that could cost money."""

from gcp import BUCKET, PROJECT, VM_NAME_PREFIX, ZONE, check_not_in_docker, gcloud, run

check_not_in_docker()

print("=== VMs ===")
gcloud(
    "compute", "instances", "list",
    f"--filter=name~^{VM_NAME_PREFIX}",
    "--format=table(name,zone,machineType.basename(),status,creationTimestamp)",
    check=False,
)

print("\n=== Disks (orphaned or attached) ===")
gcloud(
    "compute", "disks", "list",
    f"--filter=name~^{VM_NAME_PREFIX}",
    "--format=table(name,zone,sizeGb,status,users.basename())",
    check=False,
)

print(f"\n=== GCS: {BUCKET} ===")
run(
    ["gcloud", f"--project={PROJECT}", "storage", "du", BUCKET, "--summarize"],
    check=False,
)
