#!/usr/bin/env python3
"""Download sweep results from GCS.

Usage:
    python scripts/pull_results.py              # List available runs
    python scripts/pull_results.py 0309-1422    # Pull to <repo>/results/sweep-0309-1422/
    python scripts/pull_results.py 0309-1422 -d /tmp/out  # Pull to /tmp/out/
"""

import argparse
import subprocess
from pathlib import Path

from gcp import BUCKET, check_not_in_docker, run

check_not_in_docker()


def git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return Path(result.stdout.strip())


parser = argparse.ArgumentParser(description="Pull sweep results from GCS.")
parser.add_argument(
    "run_id",
    nargs="?",
    default=None,
    help="Run ID to pull (e.g. 0309-1422). Omit to list available runs.",
)
parser.add_argument(
    "-d", "--dest",
    default=None,
    help="Destination directory. Default: <repo>/results/sweep-<run-id>/",
)
args = parser.parse_args()

if args.run_id is None:
    print(f"Available runs in {BUCKET}:\n")
    run(["gcloud", "storage", "ls", f"{BUCKET}/"], check=False)
else:
    if args.dest:
        dest = Path(args.dest)
    else:
        dest = git_root() / "results" / f"sweep-{args.run_id}"
    dest.mkdir(parents=True, exist_ok=True)
    source = f"{BUCKET}/sweep-{args.run_id}/"
    print(f"Pulling {source} -> {dest}/")
    run(["gcloud", "storage", "rsync", source, str(dest), "--recursive"])
    print(f"Done. Results in {dest}/")
