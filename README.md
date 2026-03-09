# ml-reentry-serving
A learning exercise for ML infrastructure, including inference serving, performance analysis, and workflows.

## Basic remote architecture

```
WSL host                    GCP
─────────                   ───
launch_cloud.py ──SSH──→ VM
                          ├─ sweep_runner.sh
                          │   ├─ git clone
                          │   ├─ docker build
                          │   └─ docker run
                          │       ├─ cloud-entrypoint.sh
                          │       └─ run-sweep.py → MLflow
                          └─ gcloud storage upload ──→ GCS bucket

pull_results.py ←─gcloud storage──────────── GCS bucket
teardown_cloud.py ──delete──→ VM
```

## GCP resources

```
GCP Project: research-489502
├── GCS: gs://research-489502-reentry-vllm
│   └── sweep-<run-id>/          ← MLflow results per run
│
└── Compute Engine
    └── reentry-vllm-sweep-<run-id>
        ├── Image: Deep Learning VM (common-cu128-ubuntu-2204-nvidia-570)
        ├── Machine: e2-medium (test) | a2-highgpu-1g (A100)
        ├── Disk: 200GB boot
        ├── Scopes: storage-rw
        └── Container (built from .devcontainer/Dockerfile)
            ├── /x/workspace     ← cloned repo at pinned commit
            └── /results/mlflow  ← sweep output → uploaded to GCS
```
