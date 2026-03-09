#!/usr/bin/env bash
set -euo pipefail

# Symlink shared Claude Code config into ~/.claude
# (skip projects/ — it gets special handling below)
mkdir -p "$HOME/.claude/projects"
for f in /mnt/claude/config/* /mnt/claude/config/.*; do
  name=$(basename "$f")
  case "$name" in .|..|projects) continue;; esac
  ln -sfn "$f" "$HOME/.claude/$name"
done

# Symlink container project key → host project directory.
# $PWD is the container workspace folder (set by postStartCommand context).
# Claude Code will derive its project key from this same $PWD.
host_key=$(echo "$HOST_WORKSPACE_FOLDER" | tr '/' '-')
container_key=$(echo "$PWD" | tr '/' '-')
ln -sfn "/mnt/claude/projects/$host_key" "$HOME/.claude/projects/$container_key"

# Write path-map for cross-environment path translation.
# This file persists on the host (via the symlink) and tells host-side
# Claude sessions how to interpret container paths in shared history.
mkdir -p "$HOME/.claude/projects/$container_key/memory"
cat > "$HOME/.claude/projects/$container_key/memory/path-map.md" << EOF
Path equivalence: ${HOST_WORKSPACE_FOLDER} (host) = ${PWD} (container).
These refer to the same workspace. Translate paths accordingly when reading
session history or file references from the other environment.
EOF
