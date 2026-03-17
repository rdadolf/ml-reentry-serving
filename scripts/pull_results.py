#!/usr/bin/env python3
"""Download sweep results from GCS.

Usage:
    python scripts/pull_results.py                              # List available sweeps
    python scripts/pull_results.py vllm-sweep-0309-1422         # Pull to <repo>/results/vllm-sweep-0309-1422/
    python scripts/pull_results.py vllm-sweep-0309-1422 -d /tmp # Pull to /tmp/
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
    "sweep_name",
    nargs="?",
    default=None,
    help="Sweep name to pull (e.g. vllm-sweep-0309-1422). Omit to list available.",
)
parser.add_argument(
    "-d", "--dest",
    default=None,
    help="Destination directory. Default: <repo>/results/<sweep-name>/",
)
args = parser.parse_args()

if args.sweep_name is None:
    print(f"Available sweeps in {BUCKET}:\n")
    run(["gcloud", "storage", "ls", f"{BUCKET}/"], check=False)
else:
    if args.dest:
        dest = Path(args.dest)
    else:
        dest = git_root() / "results" / args.sweep_name
    dest.mkdir(parents=True, exist_ok=True)
    source = f"{BUCKET}/{args.sweep_name}/"
    print(f"Pulling {source} -> {dest}/")
    run(["gcloud", "storage", "rsync", source, str(dest), "--recursive"])
    print(f"Done. Results in {dest}/")
