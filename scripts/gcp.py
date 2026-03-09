"""Shared GCP configuration and helpers for sweep infrastructure scripts.

All scripts that interact with GCP import from here to keep project IDs,
zones, bucket names, and common operations in one place.
"""

import os
import subprocess
import sys
from pathlib import Path

# ── GCP project settings ──────────────────────────────────────────────

PROJECT = "research-489502"
ZONE = "us-west1-c"
BUCKET = "gs://research-489502-reentry-vllm"
VM_NAME_PREFIX = "reentry-vllm-sweep"

# ── Machine types ─────────────────────────────────────────────────────

CPU_MACHINE_TYPE = "e2-medium"
GPU_MACHINE_TYPE = "a2-highgpu-1g"  # A100 40GB

# ── VM image ──────────────────────────────────────────────────────────
# Deep Learning VM with CUDA drivers, Docker, and nvidia-container-toolkit
# pre-installed.

VM_IMAGE_FAMILY = "common-cu124-debian-11"
VM_IMAGE_PROJECT = "deeplearning-platform-release"

BOOT_DISK_SIZE_GB = 200

# ── Helpers ───────────────────────────────────────────────────────────


def check_not_in_docker():
    """Exit if running inside a Docker container."""
    if Path("/.dockerenv").exists():
        sys.exit("ERROR: This script must run from WSL, not inside a container.")


def run(cmd: list[str], *, check: bool = True, capture: bool = False, **kwargs):
    """Run a subprocess, printing the command for visibility."""
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, **kwargs)


def gcloud(*args: str, check: bool = True, capture: bool = False):
    """Run a gcloud command with the project flag."""
    return run(["gcloud", f"--project={PROJECT}", *args], check=check, capture=capture)


def read_token_file(path: str) -> dict[str, str]:
    """Read a Docker-style env-file (KEY=VALUE per line) and return as a dict."""
    result = {}
    p = Path(path).expanduser()
    if not p.exists():
        return result
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def require_env(name: str, fallback_file: str | None = None) -> str:
    """Get an env var, optionally falling back to a token file."""
    value = os.environ.get(name)
    if value:
        return value
    if fallback_file:
        tokens = read_token_file(fallback_file)
        value = tokens.get(name)
    if not value:
        hint = f" (or set it in {fallback_file})" if fallback_file else ""
        sys.exit(f"ERROR: {name} is required{hint}.")
    return value


def git_info() -> tuple[str, str, str]:
    """Return (branch, commit_sha, remote_url) for the current repo."""
    branch = run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True
    ).stdout.strip()
    commit = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    remote = run(
        ["git", "remote", "get-url", "origin"], capture=True
    ).stdout.strip()
    return branch, commit, remote


def git_is_clean() -> bool:
    """True if the working tree has no uncommitted changes."""
    result = run(["git", "status", "--porcelain"], capture=True)
    return result.stdout.strip() == ""


def https_clone_url(remote_url: str, token: str) -> str:
    """Convert a git remote URL to HTTPS with embedded token for auth."""
    # Handle SSH-style URLs: git@github.com:user/repo.git
    if remote_url.startswith("git@"):
        host_path = remote_url.split("git@")[1]
        host, _, path = host_path.partition(":")
        return f"https://{token}@{host}/{path}"
    # Handle HTTPS URLs: https://github.com/user/repo.git
    if remote_url.startswith("https://"):
        # Strip any existing credentials
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(remote_url)
        return urlunparse(parsed._replace(netloc=f"{token}@{parsed.hostname}"))
    return remote_url
