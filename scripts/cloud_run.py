#!/usr/bin/env python3
"""Run a sweep on an existing GCP VM.

The VM must already be provisioned and set up (via cloud_launch.py).
Each invocation generates a fresh run ID and uploads results to GCS.

Usage:
    python scripts/cloud_run.py <vm-name>                    # Run sweep
    python scripts/cloud_run.py <vm-name> -- --config x.yaml # With sweep args
    python scripts/cloud_run.py <vm-name> --cleanup          # Stop VM after
    python scripts/cloud_run.py <vm-name> --delete           # Delete VM after
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

    # Split on -- to separate our args from sweep pass-through args
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        idx = argv.index("--")
        passthrough = argv[idx + 1 :]
        argv = argv[:idx]

    parser = argparse.ArgumentParser(description="Run a sweep on an existing GCP VM.")
    parser.add_argument("name", help="VM name")
    parser.add_argument("--zone", default=ZONE, help=f"GCP zone (default: {ZONE})")
    parser.add_argument("--wait", action="store_true", help="Run in foreground, streaming output")
    parser.add_argument("--cleanup", action="store_true", help="Stop VM after sweep")
    parser.add_argument("--delete", action="store_true", help="Delete VM after sweep (implies --cleanup)")
    args = parser.parse_args(argv)

    if args.delete:
        args.cleanup = True

    return args, passthrough


def main():
    check_not_in_docker()
    args, passthrough = parse_args()

    hf_token = require_env("HF_TOKEN", "~/.hftoken")

    # Determine after-run behavior
    if args.delete:
        after_run = "delete"
    elif args.cleanup:
        after_run = "stop"
    else:
        after_run = "none"

    # Generate a fresh run ID for this sweep
    run_id = datetime.now().strftime("%m%d-%H%M")

    # Get branch/commit from the repo on the VM
    branch = ssh_to_vm(
        args.name, args.zone,
        "cd ~/repo && git rev-parse --abbrev-ref HEAD",
    ).stdout.strip()
    commit = ssh_to_vm(
        args.name, args.zone,
        "cd ~/repo && git rev-parse --short HEAD",
    ).stdout.strip()

    print(f"\n{'='*60}")
    print(f"  VM:        {args.name}")
    print(f"  Branch:    {branch}")
    print(f"  Commit:    {commit}")
    print(f"  Run ID:    {run_id}")
    print(f"  After run: {after_run}")
    if passthrough:
        print(f"  Sweep args: {' '.join(passthrough)}")
    print(f"{'='*60}\n")

    # Upload the run script
    runner_script = SCRIPTS_DIR / "run_on_vm.sh"
    scp_to_vm(args.name, args.zone, str(runner_script), "run_on_vm.sh")
    ssh_to_vm(args.name, args.zone, "chmod +x run_on_vm.sh")

    # Build the env + command
    sweep_args = " ".join(shlex.quote(a) for a in passthrough)
    env_vars = " ".join([
        f"HF_TOKEN={shlex.quote(hf_token)}",
        f"BUCKET={shlex.quote(BUCKET)}",
        f"RUN_ID={shlex.quote(run_id)}",
        f"BRANCH={shlex.quote(branch)}",
        f"COMMIT={shlex.quote(commit)}",
        f"AFTER_RUN={shlex.quote(after_run)}",
        f"PROJECT={shlex.quote(PROJECT)}",
        f"VM_ZONE={shlex.quote(args.zone)}",
    ])
    cmd = f"{env_vars} ./run_on_vm.sh {sweep_args}"

    if args.wait:
        # Run in foreground, streaming output
        print("Running sweep (foreground)...\n")
        result = ssh_to_vm(
            args.name, args.zone,
            f"bash -c '{cmd}'",
            check=False, capture=False,
        )
        if result.returncode != 0:
            print(f"\nSweep FAILED (exit code {result.returncode}).")
            sys.exit(1)
        print(f"\nSweep {run_id} completed on {args.name}.")
    else:
        # Launch detached — redirect nohup output and disown so SSH returns
        ssh_to_vm(
            args.name, args.zone,
            f"nohup bash -c '{cmd} > run.log 2>&1' > /dev/null 2>&1 & disown",
        )
        print(f"Sweep {run_id} launched on {args.name} (detached).")

    print(f"\nCheck status:")
    print(f"  python3 scripts/cloud_status.py")
    print(f"\nPull results:")
    print(f"  python3 scripts/pull_results.py --run-id {run_id}")
    if not args.wait:
        print(f"\nMonitor:")
        print(f"  gcloud compute ssh {args.name} --zone={args.zone} --project={PROJECT} --command='tail -f run.log'")
    if after_run == "none":
        print(f"\nCleanup:")
        print(f"  python3 scripts/cloud_cleanup.py {args.name}")


if __name__ == "__main__":
    main()
