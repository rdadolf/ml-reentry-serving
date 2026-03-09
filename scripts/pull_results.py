#!/usr/bin/env python3
"""Download sweep results from GCS to a local directory."""

import argparse
from pathlib import Path

from gcp import BUCKET, check_not_in_docker, run

check_not_in_docker()

parser = argparse.ArgumentParser(description="Pull sweep results from GCS.")
parser.add_argument(
    "--run-id",
    default=None,
    help="Specific run ID to pull. Omit to list available runs.",
)
parser.add_argument(
    "--dest",
    default="results",
    help="Local destination directory (default: ./results)",
)
args = parser.parse_args()

if args.run_id is None:
    print(f"Available runs in {BUCKET}:\n")
    run(["gsutil", "ls", f"{BUCKET}/"], check=False)
else:
    dest = Path(args.dest) / args.run_id
    dest.mkdir(parents=True, exist_ok=True)
    source = f"{BUCKET}/sweep-{args.run_id}/"
    print(f"Pulling {source} -> {dest}/")
    run(["gsutil", "-m", "rsync", "-r", source, str(dest)])
    print(f"Done. Results in {dest}/")
