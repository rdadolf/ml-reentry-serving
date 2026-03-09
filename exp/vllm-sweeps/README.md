# vLLM Parameter Sweeps

Sweep vLLM serving parameters (quantization, max model length, GPU memory
utilization, etc.) and log results to MLflow.

## Local

Run inside the devcontainer:

```bash
exp/vllm-sweeps/run-local.sh
```

Results go to `/home/devel/mlflow/` (bind-mounted to `~/Work/mlflow` on the
host). View with `mlflow-ui`.

## Cloud (GCP)

Run from WSL (not inside the devcontainer):

```bash
# One-time: create GCS bucket
python scripts/setup_bucket.py

# Launch sweep on a CPU test instance
python scripts/launch_cloud.py

# Launch on A100
python scripts/launch_cloud.py --gpu

# Keep VM alive after sweep (for debugging)
python scripts/launch_cloud.py --gpu --keep-alive
```

### How cloud execution works

`launch_cloud.py` creates a VM, copies `sweep_runner.sh` to it, and starts
the sweep inside a **detached `nohup` session**. This means the sweep
continues running even if your SSH connection drops — you don't need to keep
your terminal open for the duration of a multi-hour run.

To check on a running sweep:

```bash
gcloud compute ssh <vm-name> --zone=us-west1-c --project=research-489502 \
    --command='tail -f sweep.log'
```

By default the VM auto-terminates 60 seconds after the sweep finishes and
results are uploaded. Use `--keep-alive` to prevent this (e.g., to SSH in
and inspect failures).

### Retrieving results

```bash
# List available runs
python scripts/pull_results.py

# Download a specific run
python scripts/pull_results.py --run-id 0306-1430

# Teardown a kept-alive VM
python scripts/teardown_cloud.py <vm-name>
```

## Sweep configuration

Edit `sweep-config.yaml` to define the parameter grid. The sweep script
reads this file and iterates over all combinations.

## File layout

| File | Purpose |
|------|---------|
| `run-sweep.py` | Entry point — environment-agnostic |
| `run-local.sh` | Local wrapper (sets args, calls run-sweep.py) |
| `sweep-config.yaml` | Parameter grid |
