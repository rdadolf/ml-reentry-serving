#!/usr/bin/env python3
"""Stop or delete a GCP VM.

Usage:
    python scripts/cloud_cleanup.py <vm-name>            # Stop the VM
    python scripts/cloud_cleanup.py <vm-name> --delete   # Delete the VM
"""

import argparse
import sys

from gcp import PROJECT, VM_NAME_PREFIX, ZONE, check_not_in_docker, gcloud

check_not_in_docker()

parser = argparse.ArgumentParser(description="Stop or delete a sweep VM.")
parser.add_argument("name", nargs="?", default=None, help="VM name")
parser.add_argument("--zone", default=ZONE, help=f"GCP zone (default: {ZONE})")
parser.add_argument("--delete", action="store_true", help="Delete the VM (default: stop)")
args = parser.parse_args()

if args.name is None:
    print(f"ERROR: VM name is required.\n", file=sys.stderr)
    print(f"Matching instances:", file=sys.stderr)
    gcloud(
        "compute", "instances", "list",
        f"--filter=name~^{VM_NAME_PREFIX}",
        f"--zones={args.zone}",
        "--format=table(name,status,creationTimestamp)",
    )
    sys.exit(1)

if args.delete:
    gcloud(
        "compute", "instances", "delete", args.name,
        f"--zone={args.zone}",
        "--quiet",
    )
    print(f"Deleted {args.name}.")
else:
    gcloud(
        "compute", "instances", "stop", args.name,
        f"--zone={args.zone}",
    )
    print(f"Stopped {args.name}.")
    print(f"Note: Stopped VMs still incur disk costs. Use --delete to remove entirely.")
