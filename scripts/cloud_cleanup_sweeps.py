#!/usr/bin/env python3
"""Delete sweep run data from GCS by run ID.

Usage:
    python scripts/cloud_cleanup_sweeps.py 0309-1609
    python scripts/cloud_cleanup_sweeps.py 0309-1609 0308-1422
"""

import argparse
import sys

from gcp import BUCKET, check_not_in_docker, run

check_not_in_docker()

parser = argparse.ArgumentParser(description="Delete sweep run data from GCS.")
parser.add_argument("run_ids", nargs="+", help="Run ID(s) to delete")
args = parser.parse_args()

for run_id in args.run_ids:
    prefix = f"{BUCKET}/sweep-{run_id}/"
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
