#!/usr/bin/env python3
"""Provision a GCP VM, clone the repo, and build the container image.

Optionally also runs a sweep (--run) and/or cleans up after (--cleanup/--delete).

Usage:
    python scripts/cloud_launch.py                                    # Provision only
    python scripts/cloud_launch.py --gpu                              # Provision with GPU
    python scripts/cloud_launch.py --run                              # Provision + run sweep
    python scripts/cloud_launch.py --run --config x.yaml              # Custom config
    python scripts/cloud_launch.py --run --delete                     # Fire-and-forget
    python scripts/cloud_launch.py --run --sweep-name vllm-sweep-0316 # Resume
"""

import argparse
import shlex
import subprocess
import sys

from gcp import (
    BOOT_DISK_SIZE_GB,
    BUCKET,
    CPU_MACHINE_TYPE,
    GPU_MACHINE_TYPE,
    IMAGE,
    PROJECT,
    REGISTRY_LOCATION,
    SCRIPTS_DIR,
    VM_IMAGE_FAMILY,
    VM_NAME_PREFIX,
    ZONE,
    check_not_in_docker,
    create_instance,
    generate_vm_name,
    gcloud,
    git_info,
    git_is_clean,
    https_clone_url,
    image_content_hash,
    image_tag,
    require_env,
    scp_to_vm,
    ssh_to_vm,
    wait_for_ssh,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Provision a GCP VM and build the sweep container.")
    parser.add_argument("--gpu", action="store_true", help="Use A100 GPU instance")
    parser.add_argument("--vm-name", default=None, help=f"VM name (default: {VM_NAME_PREFIX}-<hex>)")
    parser.add_argument("--machine-type", default=None, help="Override machine type")
    parser.add_argument("--zone", default=ZONE, help=f"GCP zone (default: {ZONE})")
    parser.add_argument("--git-force", action="store_true", help="Skip clean-git check")
    parser.add_argument("--run", action="store_true", help="Also run a sweep after setup")
    parser.add_argument("--cleanup", action="store_true", help="Stop VM after run")
    parser.add_argument("--delete", action="store_true", help="Delete VM after run (implies --cleanup)")
    parser.add_argument("--sweep-name", default=None,
                        help="Sweep name (for resume). Auto-generated if not specified.")
    parser.add_argument("--config", default=None,
                        help="Path to sweep config YAML (on the VM)")
    args = parser.parse_args()

    if args.delete:
        args.cleanup = True
    if args.cleanup and not args.run:
        parser.error("--cleanup/--delete require --run")

    return args


def main():
    check_not_in_docker()
    args = parse_args()

    # ── Validate ──────────────────────────────────────────────────────
    if not args.git_force and not git_is_clean():
        sys.exit("ERROR: Uncommitted changes. Commit or stash, or use --git-force.")

    # Check that the container image has been pushed for the current file state
    expected_tag = image_content_hash()
    full_image_tag = image_tag()
    result = gcloud(
        "artifacts", "docker", "tags", "list",
        f"{IMAGE}",
        "--format=value(tag)",
        check=False, capture=True,
    )
    if expected_tag not in (result.stdout or ""):
        sys.exit(
            f"ERROR: No image in Artifact Registry for current file state (tag {expected_tag}).\n"
            f"Run: python scripts/cloud_push_image.py"
        )
    print(f"Image tag {expected_tag} found in registry.")

    branch, commit, remote_url = git_info()
    gh_token = require_env("GH_TOKEN", "~/.ghtoken")
    hf_token = require_env("HF_TOKEN", "~/.hftoken")
    clone_url = https_clone_url(remote_url, gh_token)

    # ── Resolve instance config ───────────────────────────────────────
    machine_type = args.machine_type or (GPU_MACHINE_TYPE if args.gpu else CPU_MACHINE_TYPE)
    name = args.vm_name or generate_vm_name()
    zone = args.zone

    print(f"\n{'='*60}")
    print(f"  Branch:       {branch}")
    print(f"  Commit:       {commit[:12]}")
    print(f"  VM:           {name}")
    print(f"  Machine type: {machine_type}")
    print(f"  GPU:          {'yes' if args.gpu else 'no (CPU test)'}")
    print(f"  Image:        {full_image_tag}")
    print(f"  Bucket:       {BUCKET}")
    if args.run:
        after = "delete" if args.delete else ("stop" if args.cleanup else "none")
        print(f"  Run sweep:    yes (after: {after})")
    print(f"{'='*60}\n")

    # ── Create VM ─────────────────────────────────────────────────────
    create_instance(name, zone, machine_type, args.gpu)
    wait_for_ssh(name, zone)

    # ── Upload and run setup ──────────────────────────────────────────
    setup_script = SCRIPTS_DIR / "vm_setup.sh"
    scp_to_vm(name, zone, str(setup_script), "vm_setup.sh")
    ssh_to_vm(name, zone, "chmod +x vm_setup.sh")

    exports = " && ".join([
        f"export CLONE_URL={shlex.quote(clone_url)}",
        f"export BRANCH={shlex.quote(branch)}",
        f"export COMMIT={shlex.quote(commit)}",
        f"export IMAGE_TAG={shlex.quote(full_image_tag)}",
    ])
    ssh_to_vm(name, zone, f"{exports} && ./vm_setup.sh", capture=False)

    print(f"\nVM {name} is ready (image built).")

    # ── Optionally run sweep ──────────────────────────────────────────
    if args.run:
        print("Starting sweep...")
        cmd = [sys.executable, str(SCRIPTS_DIR / "cloud_run.py"), name]
        if args.delete:
            cmd.append("--delete")
        elif args.cleanup:
            cmd.append("--cleanup")
        if args.sweep_name:
            cmd += ["--sweep-name", args.sweep_name]
        if args.config:
            cmd += ["--config", args.config]
        subprocess.run(cmd, check=True)
    print("\nDone.")

if __name__ == "__main__":
    main()
