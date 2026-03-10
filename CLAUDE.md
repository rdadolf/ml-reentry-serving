# Claude Code Instructions

## Git Policy
Do NOT run any git commands (commit, push, pull, checkout, branch, add, reset, rebase, merge, stash, tag, etc.). The user manages all git operations manually. This applies even if explicitly asked — confirm with the user first if a request seems to require git.

## Container Image Hash

The cloud infrastructure uses a content hash of image-affecting files to tag Docker images in Artifact Registry. The hash is computed in `gcp.py` over:
- `.devcontainer/Dockerfile`
- `pyproject.toml`
- `uv.lock`

If you add a new file that is COPYed into the Dockerfile or otherwise affects the built image, you MUST add it to `IMAGE_HASH_FILES` in `gcp.py` so the staleness check in `cloud_launch.py` remains accurate.

