#!/usr/bin/env python3
"""Build the sweep container image and push it to Artifact Registry.

Run this whenever the Dockerfile, pyproject.toml, or uv.lock changes.
The image is tagged with a content hash of those files so VMs always
pull the exact version that matches the current repo state.

Usage:
    python scripts/cloud_push_image.py
"""

from gcp import (
    IMAGE,
    REGISTRY_LOCATION,
    REPO_ROOT,
    check_not_in_docker,
    image_content_hash,
    run,
)

check_not_in_docker()

tag = image_content_hash()
full_tag = f"{IMAGE}:{tag}"
latest_tag = f"{IMAGE}:latest"

print(f"Image tag: {tag}")
print(f"Building {full_tag} ...")

# Configure Docker auth for Artifact Registry
run(["gcloud", "auth", "configure-docker", f"{REGISTRY_LOCATION}-docker.pkg.dev", "--quiet"])

# Build
run([
    "docker", "build",
    "-f", str(REPO_ROOT / ".devcontainer" / "Dockerfile"),
    "-t", full_tag,
    "-t", latest_tag,
    str(REPO_ROOT),
])

# Push both tags
print(f"Pushing {full_tag} ...")
run(["docker", "push", full_tag])
print(f"Pushing {latest_tag} ...")
run(["docker", "push", latest_tag])

print(f"\nDone. Image pushed as:")
print(f"  {full_tag}")
print(f"  {latest_tag}")
