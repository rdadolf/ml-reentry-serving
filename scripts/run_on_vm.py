#!/usr/bin/env python3
"""Run on the GCP VM (not inside a container).

Executes a sweep inside the pre-built container.
All experiment data goes directly to the MLflow tracking server.

Expected environment variables (set by cloud_run.py):
    HF_TOKEN     — Hugging Face token for model downloads
    BUCKET       — GCS bucket (gs://...)
    SWEEP_NAME   — Sweep identifier (e.g. vllm-sweep-0316-1430)
    BRANCH       — Git branch (for metadata)
    COMMIT       — Commit SHA (for metadata)
    AFTER_RUN    — "none", "stop", or "delete"
    PROJECT      — GCP project ID (needed for self-delete)
    VM_ZONE      — GCP zone (needed for self-delete)

Optional CLI arguments:
    --config PATH   Path to sweep config YAML (inside container)
"""

import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ── Environment ──────────────────────────────────────────────────────

def env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"ERROR: {name} is required.")
    return val


HF_TOKEN = env("HF_TOKEN")
BUCKET = env("BUCKET")
SWEEP_NAME = env("SWEEP_NAME")
BRANCH = env("BRANCH")
COMMIT = env("COMMIT")
AFTER_RUN = env("AFTER_RUN")
PROJECT = env("PROJECT")
VM_ZONE = env("VM_ZONE")

REPO_DIR = Path.home() / "repo"
GCS_MLFLOW_SERVER = f"{BUCKET}/mlflow-server"
VM_NAME = socket.gethostname()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run sweep on VM.")
    parser.add_argument("--config", default=None,
                        help="Path to sweep config YAML (inside container)")
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────

def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kwargs)


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and (
        subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    )


def upload_run_log():
    log = Path.home() / "run.log"
    if log.exists():
        run(
            ["gcloud", "storage", "cp", str(log),
             f"{BUCKET}/{SWEEP_NAME}/run.log"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        )


def read_mlflow_uri() -> str:
    """Read the MLflow tracking URI from GCS."""
    result = run(
        ["gcloud", "storage", "cat", GCS_MLFLOW_SERVER],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        sys.exit(
            f"ERROR: Could not read MLflow server URI from {GCS_MLFLOW_SERVER}.\n"
            "Run: python scripts/mlflow.py start"
        )
    return result.stdout.strip()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=== run_on_vm.py ===")
    print(f"Sweep:     {SWEEP_NAME}")
    print(f"After run: {AFTER_RUN}")
    print(datetime.now())

    # Discover MLflow tracking server
    mlflow_uri = read_mlflow_uri()
    print(f"MLflow tracking: {mlflow_uri}")

    # Run sweep in container
    gpu = has_gpu()
    print(f"--- Running sweep ({'GPU' if gpu else 'CPU'}) ---")

    docker_cmd = ["sudo", "docker", "run", "--rm"]
    if gpu:
        docker_cmd += ["--gpus", "all"]
    vm_user_home = Path.home()
    docker_cmd += [
        "-v", f"{REPO_DIR}:/x/workspace",
        "-v", f"{vm_user_home}/.mlflow:/home/devel/.mlflow:ro",
        "-e", f"HF_TOKEN={HF_TOKEN}",
        "-e", f"MLFLOW_TRACKING_URI={mlflow_uri}",
        "-e", f"BRANCH={BRANCH}",
        "-e", f"COMMIT={COMMIT}",
        "sweep:latest",
        "bash", "/x/workspace/.devcontainer/cloud-entrypoint.sh",
        "--name", SWEEP_NAME,
        "--resume",
    ]
    if args.config:
        docker_cmd += ["--config", args.config]

    result = run(docker_cmd, check=False)
    if result.returncode != 0:
        upload_run_log()
        sys.exit(result.returncode)

    print("=== Sweep complete ===")
    print(datetime.now())

    # Post-run VM lifecycle
    if AFTER_RUN in ("stop", "delete"):
        print(f"{AFTER_RUN.title()}ing VM in 60s...")
        time.sleep(60)

    if AFTER_RUN == "stop":
        run(["sudo", "shutdown", "-h", "now"], check=False)
    elif AFTER_RUN == "delete":
        run([
            "gcloud", "compute", "instances", "delete", VM_NAME,
            f"--zone={VM_ZONE}", f"--project={PROJECT}", "--quiet",
        ], check=False)
    else:
        print("VM left running.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        upload_run_log()
        raise
