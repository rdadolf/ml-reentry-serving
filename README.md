# ml-reentry-serving
A learning exercise for ML infrastructure, including inference serving, performance analysis, and workflows.

## Basic remote architecture

```
WSL host                         GCP
─────────                        ───
cloud_launch.py ──SSH──→ VM
  ├─ vm_setup.sh                  ├─ git clone
  │                               └─ docker build
  └─ (optional --run)
      └─ cloud_run.py ──SSH──→ VM
           └─ run_on_vm.sh        ├─ docker run
                                  │   ├─ cloud-entrypoint.sh
                                  │   └─ run-sweep.py → MLflow
                                  └─ gcloud storage upload → GCS

pull_results.py ←─gcloud storage────────── GCS bucket
cloud_cleanup.py ──stop/delete──→ VM
```

## GCP resources

```
GCP Project: research-489502
├── GCS: gs://research-489502-reentry-vllm
│   └── sweep-<MMDD-HHMM>/      ← MLflow results per run
│
└── Compute Engine
    └── reentry-vllm-<hex>
        ├── Image: Deep Learning VM (common-cu128-ubuntu-2204-nvidia-570)
        ├── Machine: e2-medium (test) | a2-highgpu-1g (A100)
        ├── Disk: 200GB boot
        ├── Scopes: storage-rw, compute-rw
        └── Container (built from .devcontainer/Dockerfile)
            ├── /x/workspace     ← cloned repo at pinned commit
            └── /results/mlflow  ← sweep output → uploaded to GCS
```
