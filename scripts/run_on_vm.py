#!/usr/bin/env python3
"""Run on the GCP VM (not inside a container).

Executes a sweep inside the pre-built container, uploads results to
GCS, and optionally stops or deletes the VM.

Expected environment variables (set by cloud_run.py):
    HF_TOKEN     — Hugging Face token for model downloads
    BUCKET       — GCS bucket (gs://...)
    RUN_ID       — Identifier for this sweep run (MMDD-HHMM)
    BRANCH       — Git branch (for metadata)
    COMMIT       — Commit SHA (for metadata)
    AFTER_RUN    — "none", "stop", or "delete"
    PROJECT      — GCP project ID (needed for self-delete)
    VM_ZONE      — GCP zone (needed for self-delete)

Any additional arguments are passed through to run-sweep.py.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ── Environment ──────────────────────────────────────────────────────

def env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"ERROR: {name} is required.")
    return val


HF_TOKEN = env("HF_TOKEN")
BUCKET = env("BUCKET")
RUN_ID = env("RUN_ID")
BRANCH = env("BRANCH")
COMMIT = env("COMMIT")
AFTER_RUN = env("AFTER_RUN")
PROJECT = env("PROJECT")
VM_ZONE = env("VM_ZONE")

REPO_DIR = Path.home() / "repo"
RESULTS_DIR = Path.home() / "results" / RUN_ID
STATUS_PATH = f"{BUCKET}/sweep-{RUN_ID}/status.json"
VM_NAME = socket.gethostname()
PASSTHROUGH_ARGS = sys.argv[1:]


# ── Helpers ──────────────────────────────────────────────────────────

def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kwargs)


def write_status(status: str, error: str = ""):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = {
        "status": status,
        "run_id": RUN_ID,
        "vm": VM_NAME,
        "branch": BRANCH,
        "commit": COMMIT,
        "timestamp": ts,
    }
    if error:
        data["error"] = error
    try:
        run(
            ["gcloud", "storage", "cp", "-", STATUS_PATH],
            input=json.dumps(data), text=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and (
        subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    )


def upload_run_log():
    log = Path.home() / "run.log"
    if log.exists():
        run(
            ["gcloud", "storage", "cp", str(log), f"{BUCKET}/sweep-{RUN_ID}/run.log"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=== run_on_vm.py ===")
    print(f"Run ID:    {RUN_ID}")
    print(f"After run: {AFTER_RUN}")
    print(datetime.now())

    # Set up results directories
    (RESULTS_DIR / "mlflow").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.chmod(0o777)
    (RESULTS_DIR / "mlflow").chmod(0o777)

    write_status("running")

    # Run sweep in container
    gpu = has_gpu()
    print(f"--- Running sweep ({'GPU' if gpu else 'CPU'}) ---")

    docker_cmd = ["sudo", "docker", "run", "--rm"]
    if gpu:
        docker_cmd += ["--gpus", "all"]
    docker_cmd += [
        "-v", f"{REPO_DIR}:/x/workspace",
        "-v", f"{RESULTS_DIR}:/results",
        "-e", f"HF_TOKEN={HF_TOKEN}",
        "-e", "MLFLOW_TRACKING_URI=sqlite:////results/mlflow/mlflow.db",
        "-e", "MLFLOW_DEFAULT_ARTIFACT_ROOT=/results/mlflow/artifacts",
        "-e", f"RUN_ID={RUN_ID}",
        "-e", f"BRANCH={BRANCH}",
        "-e", f"COMMIT={COMMIT}",
        "sweep:latest",
        "bash", "/x/workspace/.devcontainer/cloud-entrypoint.sh",
        *PASSTHROUGH_ARGS,
    ]

    result = run(docker_cmd, check=False)
    if result.returncode != 0:
        upload_run_log()
        error_detail = f"exit code {result.returncode}"
        log = Path.home() / "run.log"
        if log.exists():
            last_line = log.read_text().rstrip().rsplit("\n", 1)[-1]
            if last_line:
                error_detail = last_line
        write_status("failed", error_detail)
        sys.exit(result.returncode)

    # Upload results
    print("--- Uploading results ---")
    run([
        "gcloud", "storage", "rsync",
        str(RESULTS_DIR), f"{BUCKET}/sweep-{RUN_ID}/",
        "--recursive",
    ], check=True)
    print(f"Results uploaded to {BUCKET}/sweep-{RUN_ID}/")

    write_status("complete")
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
        write_status("failed", str(exc))
        raise
