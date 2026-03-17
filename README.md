# ml-reentry-serving
A learning exercise for ML infrastructure, including inference serving, performance analysis, and workflows.

## Architecture

Both local (devcontainer) and cloud (GCP VM) sweeps push results to a
long-running MLflow tracking server on GCP.

```
WSL host                              GCP
─────────                             ───
                               ┌───────────────────┐
                               │ reentry-mlflow    │
                               │ (persistent VM)   │
                               │ MLflow :5000      │
                               │ SQLite → GCS bkup │
                               └──────┬────────────┘
                                      │
              ┌───────────────────────┤  
              │                       │  ← spawned via cloud_launch.py
              │                       │    invoked via cloud_run.py
              │                       │
  Local devcontainer            Sweep VM (reentry-vllm-<hex>)
  ┌─────────────────┐           ┌──────────────────────────┐
  │ run-local.sh    │           │ run_on_vm.py             │
  │ └─ run-sweep.py ├──┐        │ └─ docker run            │
  └─────────────────┘  │        │   └─ cloud-entrypoint.sh │
                       │        │     └─ run-sweep.py      │
              MLflow ←─┘        └──────────┬───────────────┘
                                           │
                                           ▼
                                    MLflow, GCS: run.log
```

### MLflow server management

```
python scripts/mlflow.py create --new-credentials   # one-time setup
python scripts/mlflow.py start                       # start / refresh IP
python scripts/mlflow.py status                      # health check
python scripts/mlflow.py stop                        # stop (preserves disk)
```

The tracking URI is published to GCS (`gs://…/mlflow-server`) so cloud VMs
discover it automatically. Locally, `~/.mlflow/server` is bind-mounted into
the devcontainer.

## GCP resources

```
GCP Project: research-489502
│
├── GCS: gs://research-489502-reentry-vllm
│   ├── mlflow-server              ← tracking URI (IP:port, read by VMs)
│   ├── mlflow-backup/mlflow.db    ← SQLite backup (every 6h + on stop)
│   └── vllm-sweep-<MMDD-HHMM>/   ← run.log per sweep
│
├── Artifact Registry: us-west1-docker.pkg.dev/.../reentry-vllm/sweep:<hash>
│   └── Container images tagged by content hash of Dockerfile + deps
│
├── Persistent Disk: reentry-vllm-docker
│   └── Docker layer cache (survives VM deletion, shared across sweep VMs)
│
└── Compute Engine
    ├── reentry-mlflow  (persistent, e2-small)
    │   └── MLflow tracking server with basic-auth, systemd service
    │
    └── reentry-vllm-<hex>  (ephemeral sweep VMs)
        ├── Image: Deep Learning VM (common-cu128-ubuntu-2204-nvidia-570)
        ├── Machine: n2-standard-8 (CPU test) | g2-standard-8 (L4 GPU)
        ├── Disk: 200GB boot + docker-cache
        ├── Scopes: storage-rw, compute-rw
        └── Container (pulled from Artifact Registry)
            └── /x/workspace  ← cloned repo at pinned commit
```
