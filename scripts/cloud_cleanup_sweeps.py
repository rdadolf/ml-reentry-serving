#!/usr/bin/env python3
"""Delete sweep data from GCS by sweep name.

Usage:
    python scripts/cloud_cleanup_sweeps.py vllm-sweep-0309-1609
    python scripts/cloud_cleanup_sweeps.py vllm-sweep-0309-1609 vllm-sweep-0308-1422
"""

import argparse
import sys

from gcp import BUCKET, check_not_in_docker, run

check_not_in_docker()

parser = argparse.ArgumentParser(description="Delete sweep data from GCS.")
parser.add_argument("sweep_names", nargs="+", help="Sweep name(s) to delete")
args = parser.parse_args()

for sweep_name in args.sweep_names:
    prefix = f"{BUCKET}/{sweep_name}/"
    print(f"Deleting {prefix} ...")
    result = run(
        ["gcloud", "storage", "rm", "--recursive", prefix],
        check=False,
    )
    if result.returncode != 0:
        print(f"  warning: failed or not found: {prefix}", file=sys.stderr)
    else:
        print(f"  deleted.")

print("Done.")
