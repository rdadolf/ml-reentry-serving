#!/usr/bin/env python3
"""Show the state of all cloud resources that could cost money."""

import json
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
    "sweeps": subprocess.Popen(
        ["gsutil", "cat", f"{BUCKET}/vllm-sweep-*/status.json"],
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
header("Sweep Runs")
stdout, _ = results["sweeps"]

sweeps = []
for line in stdout.splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        sweeps.append(json.loads(line))
    except json.JSONDecodeError:
        continue

if not sweeps:
    print("  (no sweep runs found)")
else:
    sweeps.sort(key=lambda s: s.get("timestamp", ""), reverse=True)

    print(f"  {'SWEEP':<24} {'STATUS':<10} {'VM':<20} {'BRANCH':<16} {'COMMIT':<10} {'TIMESTAMP'}")
    print(f"  {'-'*24} {'-'*10} {'-'*20} {'-'*16} {'-'*10} {'-'*22}")
    for s in sweeps:
        sweep_name = s.get("sweep_name", s.get("run_id", "?"))
        status = s.get("status", "?")
        vm = s.get("vm", "?")
        branch = s.get("branch", "?")
        commit = s.get("commit", "?")
        ts = s.get("timestamp", "?")

        print(f"  {sweep_name:<24} {status:<10} {vm:<20} {branch:<16} {commit:<10} {ts}")
        if s.get("status") == "failed" and s.get("error"):
            print(f"    error: {s['error']}")
