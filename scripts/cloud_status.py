#!/usr/bin/env python3
"""Show the state of all cloud resources that could cost money.

For running sweep VMs, also reports sweep progress from MLflow,
container health, and GPU/CPU utilization.
"""

import configparser
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from gcp import BUCKET, IMAGE, PROJECT, VM_NAME_PREFIX, ZONE, check_not_in_docker, image_content_hash

check_not_in_docker()

W = 75
MLFLOW_DIR = Path.home() / ".mlflow"


def header(title):
    s = f"=== {title} "
    print(s + "=" * (W - len(s)))


def read_mlflow_connection() -> tuple[str, str, str] | None:
    """Read MLflow URI and credentials from local config files.

    Returns (tracking_uri, username, password) or None if unavailable.
    """
    server_path = MLFLOW_DIR / "server"
    creds_path = MLFLOW_DIR / "credentials"
    if not server_path.exists() or not creds_path.exists():
        return None
    tracking_uri = server_path.read_text().strip()
    config = configparser.ConfigParser()
    config.read(creds_path)
    try:
        username = config["mlflow"]["mlflow_tracking_username"]
        password = config["mlflow"]["mlflow_tracking_password"]
    except KeyError:
        return None
    return tracking_uri, username, password


def mlflow_request(tracking_uri: str, username: str, password: str,
                   path: str, *, method: str = "GET", body: dict | None = None):
    """Make an authenticated request to the MLflow REST API."""
    import base64
    url = f"{tracking_uri}/api/2.0/mlflow/{path}"
    auth = base64.b64encode(f"{username}:{password}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = Request(url, headers=headers, data=data, method=method)
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except (URLError, json.JSONDecodeError):
        return None


def get_sweep_progress(tracking_uri, username, password, experiment_name):
    """Query MLflow for sweep progress: completed, failed, total."""
    # Find experiment by name
    data = mlflow_request(tracking_uri, username, password,
                          f"experiments/get-by-name?experiment_name={experiment_name}")
    if not data or "experiment" not in data:
        return None

    exp = data["experiment"]
    exp_id = exp["experiment_id"]
    tags = {t["key"]: t["value"] for t in exp.get("tags", [])}
    n_completed = int(tags.get("n_completed", "0"))
    n_total = int(tags.get("n_total", "0"))
    creation_time = int(exp.get("creation_time", 0)) / 1000  # ms → sec

    # Count failed runs
    search_data = mlflow_request(tracking_uri, username, password,
                                 "runs/search", method="POST", body={
                                     "experiment_ids": [exp_id],
                                     "filter": "status = 'FAILED'",
                                     "max_results": 1000,
                                 })
    n_failed = 0
    if search_data and "runs" in search_data:
        n_failed = len(search_data["runs"])

    return {
        "n_completed": n_completed,
        "n_total": n_total,
        "n_failed": n_failed,
        "creation_time": creation_time,
    }


def get_vm_sweep_info(vm_name, zone):
    """Get sweep name from VM metadata and container/system status via SSH.

    Returns (sweep_name, ssh_info_dict) where ssh_info_dict has keys:
    container_status, gpu_util, cpu_load. Returns (None, {}) on failure.
    """
    # Get metadata (no SSH needed)
    try:
        result = subprocess.run(
            ["gcloud", f"--project={PROJECT}", "compute", "instances", "describe",
             vm_name, f"--zone={zone}",
             "--format=value(metadata.items.filter(key:sweep-name).firstof(value))"],
            capture_output=True, text=True, check=True, timeout=15,
        )
        sweep_name = result.stdout.strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        sweep_name = None

    # Single SSH call for container status + system stats
    ssh_cmd = " && ".join([
        # Container status: "running", "exited", or "none"
        "if [ \"$(sudo docker ps -q --filter ancestor=sweep:latest 2>/dev/null)\" ]; then "
        "echo CONTAINER=running; "
        "elif [ \"$(sudo docker ps -aq --filter ancestor=sweep:latest 2>/dev/null)\" ]; then "
        "echo CONTAINER=exited; "
        "else echo CONTAINER=none; fi",
        # GPU utilization
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits 2>/dev/null | head -1 | "
        "awk -F', ' '{printf \"GPU_UTIL=%s GPU_MEM=%s/%sMiB\\n\", $1, $2, $3}' "
        "|| echo GPU_UTIL=N/A",
        # CPU load
        "awk '{printf \"CPU_LOAD=%s\\n\", $1}' /proc/loadavg",
    ])

    info = {}
    try:
        result = subprocess.run(
            ["gcloud", f"--project={PROJECT}", "compute", "ssh", vm_name,
             f"--zone={zone}", f"--command={ssh_cmd}",
             "--ssh-flag=-o ConnectTimeout=10",
             "--ssh-flag=-o StrictHostKeyChecking=no"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        for line in result.stdout.strip().splitlines():
            if line.startswith("CONTAINER="):
                info["container_status"] = line.split("=", 1)[1]
            elif line.startswith("GPU_UTIL="):
                parts = line.split()
                info["gpu_util"] = parts[0].split("=", 1)[1] + "%"
                if len(parts) > 1:
                    info["gpu_mem"] = parts[1].split("=", 1)[1]
            elif line.startswith("CPU_LOAD="):
                info["cpu_load"] = line.split("=", 1)[1]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    return sweep_name, info


# Launch resource queries in parallel (same as before)
procs = {
    "vms": subprocess.Popen(
        ["gcloud", f"--project={PROJECT}", "compute", "instances", "list",
         f"--filter=name~^{VM_NAME_PREFIX}",
         "--format=json(name,zone.basename(),machineType.basename(),status,creationTimestamp)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ),
    "disks": subprocess.Popen(
        ["gcloud", f"--project={PROJECT}", "compute", "disks", "list",
         f"--filter=name~^{VM_NAME_PREFIX}",
         "--format=table(name,zone,sizeGb,status,users.basename())"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ),
    "gcs": subprocess.Popen(
        ["gcloud", f"--project={PROJECT}", "storage", "du", BUCKET, "--summarize"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ),
    "images": subprocess.Popen(
        ["gcloud", f"--project={PROJECT}", "artifacts", "docker", "images", "list",
         IMAGE, "--include-tags", "--sort-by=~UPDATE_TIME",
         "--format=table(package,tags,updateTime.date(tz=UTC))"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ),
}

results = {name: proc.communicate() for name, proc in procs.items()}

# Parse VM list as JSON for structured access
vm_stdout, _ = results["vms"]
try:
    vms = json.loads(vm_stdout) if vm_stdout.strip() else []
except json.JSONDecodeError:
    vms = []

header("VMs")
if not vms:
    print("  (none)")
else:
    for vm in vms:
        name = vm.get("name", "?")
        zone = vm.get("zone", ZONE)
        machine = vm.get("machineType", "?")
        status = vm.get("status", "?")
        created = vm.get("creationTimestamp", "?")
        print(f"  {name}  {zone}  {machine}  {status}  {created}")

# For running sweep VMs, show sweep details
mlflow_conn = read_mlflow_connection()
sweep_vms = [vm for vm in vms
             if vm.get("status") == "RUNNING"
             and vm.get("name", "").startswith(VM_NAME_PREFIX)
             and vm.get("name") != "reentry-mlflow"]

for vm in sweep_vms:
    name = vm["name"]
    zone = vm.get("zone", ZONE)
    print()
    header(f"Sweep: {name}")

    sweep_name, info = get_vm_sweep_info(name, zone)

    if sweep_name:
        print(f"  Sweep name: {sweep_name}")
    else:
        print("  Sweep name: (no metadata)")

    # Container status
    container = info.get("container_status", "unknown")
    print(f"  Container:  {container}")

    # System stats
    gpu_util = info.get("gpu_util", "N/A")
    gpu_mem = info.get("gpu_mem", "")
    cpu_load = info.get("cpu_load", "N/A")
    gpu_str = f"{gpu_util} util"
    if gpu_mem:
        gpu_str += f", {gpu_mem}"
    print(f"  GPU:        {gpu_str}")
    print(f"  CPU load:   {cpu_load}")

    # MLflow progress (only if we know the sweep name)
    if sweep_name and mlflow_conn:
        tracking_uri, username, password = mlflow_conn
        progress = get_sweep_progress(tracking_uri, username, password, sweep_name)
        if progress:
            n_done = progress["n_completed"]
            n_total = progress["n_total"]
            n_failed = progress["n_failed"]

            if n_total > 0:
                pct_done = n_done / n_total * 100
                progress_str = f"{n_done}/{n_total} ({pct_done:.0f}%) complete"
            else:
                progress_str = f"{n_done}/? complete"

            if n_done > 0 and n_failed > 0:
                pct_failed = n_failed / n_done * 100
                progress_str += f" -- {n_failed}/{n_done} failed ({pct_failed:.0f}%)"

            print(f"  Progress:   {progress_str}")

            # Elapsed time
            if progress["creation_time"] > 0:
                elapsed = datetime.now(timezone.utc) - datetime.fromtimestamp(
                    progress["creation_time"], tz=timezone.utc)
                minutes = int(elapsed.total_seconds() / 60)
                if minutes >= 60:
                    print(f"  Elapsed:    {minutes // 60}h {minutes % 60}m")
                else:
                    print(f"  Elapsed:    {minutes}m")
        else:
            print(f"  Progress:   (experiment '{sweep_name}' not found in MLflow)")
    elif sweep_name:
        print("  Progress:   (MLflow credentials not available)")

print()
header("Disks (orphaned or attached)")
stdout, _ = results["disks"]
print(stdout.rstrip() if stdout.strip() else "  (none)")
header(f"GCS: {BUCKET}")
stdout, _ = results["gcs"]
print(stdout.rstrip() if stdout.strip() else "  (empty or not found)")
header(f"Container Images (current: {image_content_hash()})")
stdout, _ = results["images"]
print(stdout.rstrip() if stdout.strip() else "  (none)")
