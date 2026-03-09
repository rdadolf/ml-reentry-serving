#!/usr/bin/env python3
"""Launch a parameter sweep on a GCP VM.

Validates the local environment, creates a VM, copies the sweep runner
script, and kicks off execution in a detached session. The sweep continues
even if the SSH connection drops.

Usage:
    python scripts/launch_cloud.py                     # CPU test instance
    python scripts/launch_cloud.py --gpu               # A100 instance
    python scripts/launch_cloud.py --gpu --keep-alive  # Don't auto-terminate
"""

import argparse
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

from gcp import (
    BOOT_DISK_SIZE_GB,
    BUCKET,
    CPU_MACHINE_TYPE,
    GPU_MACHINE_TYPE,
    PROJECT,
    VM_IMAGE_FAMILY,
    VM_IMAGE_PROJECT,
    VM_NAME_PREFIX,
    ZONE,
    check_not_in_docker,
    gcloud,
    git_info,
    git_is_clean,
    https_clone_url,
    require_env,
    run,
)

SCRIPTS_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a sweep on a GCP VM.")
    parser.add_argument(
        "--gpu", action="store_true", help="Use A100 GPU instance (default: CPU test)"
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Don't auto-terminate VM after sweep completes",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=f"VM name (default: {VM_NAME_PREFIX}-NNNN)",
    )
    parser.add_argument(
        "--machine-type",
        default=None,
        help=f"Override machine type (default: {CPU_MACHINE_TYPE} or {GPU_MACHINE_TYPE})",
    )
    parser.add_argument(
        "--zone",
        default=ZONE,
        help=f"GCP zone (default: {ZONE})",
    )
    parser.add_argument(
        "--git-force",
        action="store_true",
        help="Skip the clean-git check (allow uncommitted changes)",
    )
    return parser.parse_args()


def create_instance(name: str, zone: str, machine_type: str, gpu: bool):
    """Create a GCP VM."""
    cmd = [
        "compute", "instances", "create", name,
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--image-family={VM_IMAGE_FAMILY}",
        f"--image-project={VM_IMAGE_PROJECT}",
        f"--boot-disk-size={BOOT_DISK_SIZE_GB}GB",
        "--scopes=storage-rw",
        "--metadata=install-nvidia-driver=True",
    ]
    if gpu:
        cmd += ["--accelerator=type=nvidia-tesla-a100,count=1", "--maintenance-policy=TERMINATE"]
    gcloud(*cmd)


def wait_for_ssh(name: str, zone: str, max_wait: int = 300):
    """Poll until SSH is available on the VM."""
    print(f"Waiting for SSH on {name}...", end="", flush=True)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        result = gcloud(
            "compute", "ssh", name, f"--zone={zone}",
            "--command=true",
            "--ssh-flag=-o ConnectTimeout=5",
            "--ssh-flag=-o StrictHostKeyChecking=no",
            check=False, capture=True,
        )
        if result.returncode == 0:
            print(" ready.")
            return
        print(".", end="", flush=True)
        time.sleep(10)
    sys.exit(f"\nERROR: SSH not available on {name} after {max_wait}s.")


def ssh_to_vm(name: str, zone: str, command: str, *, check: bool = True):
    """Run a command on the VM via gcloud compute ssh."""
    return gcloud(
        "compute", "ssh", name, f"--zone={zone}",
        f"--command={command}",
        check=check, capture=True,
    )


def scp_to_vm(name: str, zone: str, local_path: str, remote_path: str):
    """Copy a file to the VM."""
    gcloud(
        "compute", "scp", local_path, f"{name}:{remote_path}",
        f"--zone={zone}",
    )


def main():
    check_not_in_docker()
    args = parse_args()

    # ── Validate ──────────────────────────────────────────────────────
    if not args.git_force and not git_is_clean():
        sys.exit("ERROR: Uncommitted changes. Commit or stash, or use --git-force.")

    branch, commit, remote_url = git_info()
    gh_token = require_env("GH_TOKEN", "~/.ghtoken")
    hf_token = require_env("HF_TOKEN", "~/.hftoken")
    clone_url = https_clone_url(remote_url, gh_token)

    # ── Resolve instance config ───────────────────────────────────────
    machine_type = args.machine_type or (GPU_MACHINE_TYPE if args.gpu else CPU_MACHINE_TYPE)
    run_id = datetime.now().strftime("%m%d-%H%M")
    name = args.name or f"{VM_NAME_PREFIX}-{run_id}"
    zone = args.zone

    print(f"\n{'='*60}")
    print(f"  Branch:       {branch}")
    print(f"  Commit:       {commit[:12]}")
    print(f"  VM:           {name}")
    print(f"  Machine type: {machine_type}")
    print(f"  GPU:          {'yes' if args.gpu else 'no (CPU test)'}")
    print(f"  Keep alive:   {'yes' if args.keep_alive else 'no (auto-terminate)'}")
    print(f"  Bucket:       {BUCKET}")
    print(f"{'='*60}\n")

    # ── Create VM ─────────────────────────────────────────────────────
    create_instance(name, zone, machine_type, args.gpu)
    wait_for_ssh(name, zone)

    # ── Upload sweep runner ───────────────────────────────────────────
    runner_path = SCRIPTS_DIR / "sweep_runner.sh"
    scp_to_vm(name, zone, str(runner_path), "sweep_runner.sh")
    ssh_to_vm(name, zone, "chmod +x sweep_runner.sh")

    # ── Launch sweep (detached) ───────────────────────────────────────
    env_vars = " ".join([
        f"CLONE_URL={shlex.quote(clone_url)}",
        f"BRANCH={shlex.quote(branch)}",
        f"COMMIT={shlex.quote(commit)}",
        f"HF_TOKEN={shlex.quote(hf_token)}",
        f"BUCKET={shlex.quote(BUCKET)}",
        f"RUN_ID={shlex.quote(run_id)}",
        f"KEEP_ALIVE={'1' if args.keep_alive else '0'}",
    ])
    ssh_to_vm(
        name, zone,
        f"nohup bash -c '{env_vars} ./sweep_runner.sh > sweep.log 2>&1' &",
    )

    print(f"\nSweep launched on {name} (detached).")
    print(f"\nMonitor:")
    print(f"  gcloud compute ssh {name} --zone={zone} --project={PROJECT} --command='tail -f sweep.log'")
    print(f"\nTeardown (if --keep-alive):")
    print(f"  python scripts/teardown_cloud.py {name}")
    print(f"\nPull results:")
    print(f"  python scripts/pull_results.py --run-id {run_id}")


if __name__ == "__main__":
    main()
