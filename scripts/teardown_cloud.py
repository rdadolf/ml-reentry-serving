#!/usr/bin/env python3
"""Delete a sweep VM."""

import argparse
import sys

from gcp import PROJECT, ZONE, VM_NAME_PREFIX, check_not_in_docker, gcloud

check_not_in_docker()

parser = argparse.ArgumentParser(description="Delete a sweep VM.")
parser.add_argument(
    "name",
    nargs="?",
    default=None,
    help=f"VM name (default: list running {VM_NAME_PREFIX}-* instances)",
)
args = parser.parse_args()

if args.name is None:
    # List matching instances so the user can pick one.
    print(f"Running {VM_NAME_PREFIX}-* instances:\n")
    gcloud(
        "compute", "instances", "list",
        f"--filter=name~^{VM_NAME_PREFIX}",
        f"--zones={ZONE}",
        "--format=table(name,status,creationTimestamp)",
    )
    sys.exit(0)

gcloud(
    "compute", "instances", "delete", args.name,
    f"--zone={ZONE}",
    "--quiet",
)
print(f"Deleted {args.name}.")
