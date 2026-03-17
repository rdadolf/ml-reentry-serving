#!/usr/bin/env python3
"""Run a sweep on an existing GCP VM.

The VM must already be provisioned and set up (via cloud_launch.py).
Each invocation generates a fresh sweep name unless --sweep-name is
provided (for resuming a prior sweep).

Usage:
    python scripts/cloud_run.py <vm>                                        # New sweep
    python scripts/cloud_run.py <vm> --sweep-name vllm-sweep-0316-1430      # Resume
    python scripts/cloud_run.py <vm> --config x.yaml                        # Custom config
    python scripts/cloud_run.py <vm> --cleanup                              # Stop VM after
    python scripts/cloud_run.py <vm> --delete                               # Delete VM after
"""

import shlex
import sys
from datetime import datetime

from gcp import (
    BUCKET,
    PROJECT,
    SCRIPTS_DIR,
    VM_NAME_PREFIX,
    ZONE,
    check_not_in_docker,
    gcloud,
    require_env,
    run,
    scp_to_vm,
    ssh_to_vm,
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run a sweep on an existing GCP VM.")
    parser.add_argument("vm_name", help="VM name")
    parser.add_argument("--zone", default=ZONE, help=f"GCP zone (default: {ZONE})")
    parser.add_argument("--wait", action="store_true", help="Run in foreground, streaming output")
    parser.add_argument("--cleanup", action="store_true", help="Stop VM after sweep")
    parser.add_argument("--delete", action="store_true", help="Delete VM after sweep (implies --cleanup)")
    parser.add_argument("--sweep-name", default=None,
                        help="Sweep name (for resume). Auto-generated if not specified.")
    parser.add_argument("--config", default=None,
                        help="Path to sweep config YAML (on the VM)")
    args = parser.parse_args()

    if args.delete:
        args.cleanup = True

    return args


def main():
    check_not_in_docker()
    args = parse_args()

    hf_token = require_env("HF_TOKEN", "~/.hftoken")

    # Determine after-run behavior
    if args.delete:
        after_run = "delete"
    elif args.cleanup:
        after_run = "stop"
    else:
        after_run = "none"

    # Sweep name: provided (resume) or generated (new)
    sweep_name = args.sweep_name or f"vllm-sweep-{datetime.now().strftime('%m%d-%H%M')}"

    # Get branch/commit from the repo on the VM
    branch = ssh_to_vm(
        args.vm_name, args.zone,
        "cd ~/repo && git rev-parse --abbrev-ref HEAD",
    ).stdout.strip()
    commit = ssh_to_vm(
        args.vm_name, args.zone,
        "cd ~/repo && git rev-parse --short HEAD",
    ).stdout.strip()

    print(f"\n{'='*60}")
    print(f"  VM:        {args.vm_name}")
    print(f"  Branch:    {branch}")
    print(f"  Commit:    {commit}")
    print(f"  Sweep:     {sweep_name}")
    print(f"  After run: {after_run}")
    if args.config:
        print(f"  Config:    {args.config}")
    print(f"{'='*60}\n")

    # Upload the run script
    runner_script = SCRIPTS_DIR / "run_on_vm.py"
    scp_to_vm(args.vm_name, args.zone, str(runner_script), "run_on_vm.py")

    # Build the env + command
    env_vars = " ".join([
        f"HF_TOKEN={shlex.quote(hf_token)}",
        f"BUCKET={shlex.quote(BUCKET)}",
        f"SWEEP_NAME={shlex.quote(sweep_name)}",
        f"BRANCH={shlex.quote(branch)}",
        f"COMMIT={shlex.quote(commit)}",
        f"AFTER_RUN={shlex.quote(after_run)}",
        f"PROJECT={shlex.quote(PROJECT)}",
        f"VM_ZONE={shlex.quote(args.zone)}",
    ])
    config_arg = f" --config {shlex.quote(args.config)}" if args.config else ""
    cmd = f"{env_vars} python3 ./run_on_vm.py{config_arg}"

    if args.wait:
        # Run in foreground, streaming output
        print("Running sweep (foreground)...\n")
        result = ssh_to_vm(
            args.vm_name, args.zone,
            f"bash -c '{cmd}'",
            check=False, capture=False,
        )
        if result.returncode != 0:
            print(f"\nSweep FAILED (exit code {result.returncode}).")
            sys.exit(1)
        print(f"\nSweep {sweep_name} completed on {args.vm_name}.")
    else:
        # Launch detached — redirect nohup output and disown so SSH returns
        ssh_to_vm(
            args.vm_name, args.zone,
            f"nohup bash -c '{cmd} > run.log 2>&1' > /dev/null 2>&1 & disown",
        )
        print(f"Sweep {sweep_name} launched on {args.vm_name} (detached).")

    print(f"\nCheck status:")
    print(f"  python3 scripts/cloud_status.py")
    print(f"\nResume (if preempted):")
    print(f"  python3 scripts/cloud_run.py {args.vm_name} --sweep-name {sweep_name}")
    if not args.wait:
        print(f"\nMonitor:")
        print(f"  gcloud compute ssh {args.vm_name} --zone={args.zone} --project={PROJECT} --command='tail -f run.log'")
    if after_run == "none":
        print(f"\nCleanup:")
        print(f"  python3 scripts/cloud_cleanup.py {args.vm_name}")


if __name__ == "__main__":
    main()
