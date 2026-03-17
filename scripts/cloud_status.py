#!/usr/bin/env python3
"""Show the state of all cloud resources that could cost money."""

import subprocess

from gcp import BUCKET, IMAGE, PROJECT, VM_NAME_PREFIX, check_not_in_docker, image_content_hash

check_not_in_docker()

W = 75

def header(title):
    s = f"=== {title} "
    print(s + "=" * (W - len(s)))

# Launch all queries in parallel
procs = {
    "vms": subprocess.Popen(
        ["gcloud", f"--project={PROJECT}", "compute", "instances", "list",
         f"--filter=name~^{VM_NAME_PREFIX}",
         "--format=table(name,zone,machineType.basename(),status,creationTimestamp)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ),
    "disks": subprocess.Popen(
        ["gcloud", f"--project={PROJECT}", "compute", "disks", "list",
         f"--filter=name~^{VM_NAME_PREFIX}",
         "--format=table(name,zone,sizeGb,status,users.basename())"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ),
    "gcs": subprocess.Popen(
        ["gcloud", f"--project={PROJECT}", "storage", "du", BUCKET, "--summarize"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ),
    "images": subprocess.Popen(
        ["gcloud", f"--project={PROJECT}", "artifacts", "docker", "images", "list",
         IMAGE, "--include-tags", "--sort-by=~UPDATE_TIME",
         "--format=table(package,tags,updateTime.date(tz=UTC))"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ),
}

# Wait for all
results = {name: proc.communicate() for name, proc in procs.items()}

header("VMs")
stdout, _ = results["vms"]
print(stdout.rstrip() if stdout.strip() else "  (none)")
header("Disks (orphaned or attached)")
stdout, _ = results["disks"]
print(stdout.rstrip() if stdout.strip() else "  (none)")
header(f"GCS: {BUCKET}")
stdout, _ = results["gcs"]
print(stdout.rstrip() if stdout.strip() else "  (empty or not found)")
header(f"Container Images (current: {image_content_hash()})")
stdout, _ = results["images"]
print(stdout.rstrip() if stdout.strip() else "  (none)")
