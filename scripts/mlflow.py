#!/usr/bin/env python3
"""Manage the MLflow tracking server on GCP.

Usage:
    python scripts/mlflow.py create [--new-credentials]
    python scripts/mlflow.py start
    python scripts/mlflow.py status
    python scripts/mlflow.py stop
    python scripts/mlflow.py delete
"""

import configparser
import json
import secrets
import string
import subprocess
import sys
from pathlib import Path

from gcp import (
    BUCKET, PROJECT, ZONE, check_not_in_docker, gcloud, run,
    wait_for_ssh, ssh_to_vm,
)

check_not_in_docker()

MLFLOW_VM_NAME = "reentry-mlflow"
MLFLOW_PORT = 5000
MLFLOW_DIR = Path.home() / ".mlflow"
CREDENTIALS_PATH = MLFLOW_DIR / "credentials"
SERVER_PATH = MLFLOW_DIR / "server"
GCS_SERVER_PATH = f"{BUCKET}/mlflow-server"
GCS_BACKUP_DIR = f"{BUCKET}/mlflow-backup"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def generate_password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_credentials():
    """Generate a credentials file with random admin username and password."""
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    username = f"admin-{secrets.token_hex(3)}"
    password = generate_password()

    config = configparser.ConfigParser()
    config["mlflow"] = {
        "mlflow_tracking_username": username,
        "mlflow_tracking_password": password,
    }
    with open(CREDENTIALS_PATH, "w") as f:
        config.write(f)
    CREDENTIALS_PATH.chmod(0o600)

    print(f"Credentials written to {CREDENTIALS_PATH}")
    print(f"  username: {username}")
    print(f"  password: {password}")


def load_credentials() -> tuple[str, str]:
    """Read username and password from the credentials file."""
    if not CREDENTIALS_PATH.exists():
        sys.exit(
            f"ERROR: {CREDENTIALS_PATH} not found.\n"
            f"Run: python scripts/mlflow.py create --new-credentials"
        )
    config = configparser.ConfigParser()
    config.read(CREDENTIALS_PATH)
    try:
        username = config["mlflow"]["mlflow_tracking_username"]
        password = config["mlflow"]["mlflow_tracking_password"]
    except KeyError:
        sys.exit(f"ERROR: {CREDENTIALS_PATH} is malformed.")
    return username, password


# ---------------------------------------------------------------------------
# Server IP tracking
# ---------------------------------------------------------------------------

def write_server_uri(ip: str):
    """Write the tracking URI to local file and GCS."""
    uri = f"http://{ip}:{MLFLOW_PORT}"
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    SERVER_PATH.write_text(uri)

    # Also write to GCS so cloud VMs can discover it
    run(
        ["gcloud", f"--project={PROJECT}", "storage", "cp", "-",
         GCS_SERVER_PATH],
        input=uri, text=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    )
    print(f"  Tracking URI: {uri}")


def read_server_uri() -> str | None:
    """Read the tracking URI from local file."""
    if SERVER_PATH.exists():
        return SERVER_PATH.read_text().strip()
    return None


# ---------------------------------------------------------------------------
# VM management
# ---------------------------------------------------------------------------

def vm_exists() -> bool:
    result = gcloud(
        "compute", "instances", "describe", MLFLOW_VM_NAME,
        f"--zone={ZONE}", "--format=value(name)",
        check=False, capture=True,
    )
    return result.returncode == 0


def vm_status() -> str | None:
    """Return VM status string (RUNNING, TERMINATED, etc.) or None."""
    result = gcloud(
        "compute", "instances", "describe", MLFLOW_VM_NAME,
        f"--zone={ZONE}", "--format=value(status)",
        check=False, capture=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def vm_external_ip() -> str | None:
    result = gcloud(
        "compute", "instances", "describe", MLFLOW_VM_NAME,
        f"--zone={ZONE}",
        "--format=value(networkInterfaces[0].accessConfigs[0].natIP)",
        check=False, capture=True,
    )
    if result.returncode != 0:
        return None
    ip = result.stdout.strip()
    return ip if ip else None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_create(new_credentials: bool = False):
    """Create the MLflow server VM and configure it."""
    if new_credentials:
        generate_credentials()

    username, password = load_credentials()

    if vm_exists():
        print(f"VM {MLFLOW_VM_NAME} already exists.")
        status = vm_status()
        if status != "RUNNING":
            print(f"  Status: {status}. Run 'start' to bring it up.")
            return
    else:
        print(f"Creating VM {MLFLOW_VM_NAME}...")
        gcloud(
            "compute", "instances", "create", MLFLOW_VM_NAME,
            f"--zone={ZONE}",
            "--machine-type=e2-micro",
            "--image-family=ubuntu-2204-lts",
            "--image-project=ubuntu-os-cloud",
            "--boot-disk-size=10GB",
            "--scopes=storage-rw",
            "--provisioning-model=STANDARD",
            "--tags=mlflow-server",
        )

    # Open firewall for MLflow port
    gcloud(
        "compute", "firewall-rules", "create", "allow-mlflow",
        f"--allow=tcp:{MLFLOW_PORT}",
        "--target-tags=mlflow-server",
        "--source-ranges=0.0.0.0/0",
        "--description=Allow MLflow tracking server access",
        check=False,  # may already exist
    )

    wait_for_ssh(MLFLOW_VM_NAME, ZONE)

    # Install MLflow on the VM
    print("Installing MLflow on VM...")
    ssh_to_vm(MLFLOW_VM_NAME, ZONE,
              "sudo apt-get update -qq && "
              "sudo apt-get install -y -qq python3-pip > /dev/null 2>&1 && "
              "pip3 install --quiet mlflow")

    # Create the MLflow data directory
    ssh_to_vm(MLFLOW_VM_NAME, ZONE, "mkdir -p ~/mlflow")

    # Restore backup from GCS if it exists
    print("Checking for existing backup...")
    restore_result = gcloud(
        "storage", "cp", f"{GCS_BACKUP_DIR}/mlflow.db",
        f"/tmp/mlflow-restore.db",
        check=False, capture=True,
    )
    if restore_result.returncode == 0:
        # SCP wouldn't work here since we downloaded to local — use SSH
        ssh_to_vm(MLFLOW_VM_NAME, ZONE,
                  f"gcloud storage cp {GCS_BACKUP_DIR}/mlflow.db ~/mlflow/mlflow.db")
        print("  Restored MLflow database from backup.")

    # Create systemd service for MLflow with auth
    service_unit = f"""\
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User={ssh_to_vm(MLFLOW_VM_NAME, ZONE, "whoami").stdout.strip()}
Environment=HOME=/home/{ssh_to_vm(MLFLOW_VM_NAME, ZONE, "whoami").stdout.strip()}
ExecStart=/home/{ssh_to_vm(MLFLOW_VM_NAME, ZONE, "whoami").stdout.strip()}/.local/bin/mlflow server \
    --host 0.0.0.0 \
    --port {MLFLOW_PORT} \
    --backend-store-uri sqlite:///home/{ssh_to_vm(MLFLOW_VM_NAME, ZONE, "whoami").stdout.strip()}/mlflow/mlflow.db \
    --app-name basic-auth
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
    # Get the username once
    vm_user = ssh_to_vm(MLFLOW_VM_NAME, ZONE, "whoami").stdout.strip()

    service_unit = f"""\
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User={vm_user}
Environment=HOME=/home/{vm_user}
ExecStart=/home/{vm_user}/.local/bin/mlflow server \
    --host 0.0.0.0 \
    --port {MLFLOW_PORT} \
    --backend-store-uri sqlite:////home/{vm_user}/mlflow/mlflow.db \
    --app-name basic-auth
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
    ssh_to_vm(MLFLOW_VM_NAME, ZONE,
              f"echo '{service_unit}' | sudo tee /etc/systemd/system/mlflow.service > /dev/null")

    # Set up the admin user credentials on the VM
    # MLflow basic-auth uses a config file for the initial admin
    auth_config = f"""\
[mlflow]
default_permission = READ
admin_username = {username}
admin_password = {password}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
"""
    ssh_to_vm(MLFLOW_VM_NAME, ZONE,
              f"mkdir -p /home/{vm_user}/.mlflow && "
              f"echo '{auth_config}' > /home/{vm_user}/.mlflow/basic_auth.ini")

    # Set up SQLite backup cron (every 6 hours)
    backup_cmd = (
        f"gcloud storage cp /home/{vm_user}/mlflow/mlflow.db "
        f"{GCS_BACKUP_DIR}/mlflow.db"
    )
    ssh_to_vm(MLFLOW_VM_NAME, ZONE,
              f'(crontab -l 2>/dev/null; echo "0 */6 * * * {backup_cmd}") '
              f"| sort -u | crontab -")
    print("  SQLite backup to GCS scheduled every 6 hours.")

    # Start the service
    ssh_to_vm(MLFLOW_VM_NAME, ZONE,
              "sudo systemctl daemon-reload && "
              "sudo systemctl enable mlflow && "
              "sudo systemctl start mlflow")

    ip = vm_external_ip()
    if ip:
        write_server_uri(ip)

    print(f"\nMLflow server created and running on {MLFLOW_VM_NAME}.")


def cmd_start():
    """Start the MLflow VM (idempotent)."""
    status = vm_status()
    if status is None:
        sys.exit(f"ERROR: VM {MLFLOW_VM_NAME} does not exist. Run 'create' first.")
    if status == "RUNNING":
        print(f"VM {MLFLOW_VM_NAME} is already running.")
        ip = vm_external_ip()
        if ip:
            write_server_uri(ip)
        return

    print(f"Starting VM {MLFLOW_VM_NAME}...")
    gcloud("compute", "instances", "start", MLFLOW_VM_NAME, f"--zone={ZONE}")

    # Get the new ephemeral IP (available immediately after start returns)
    ip = vm_external_ip()
    if ip:
        write_server_uri(ip)
    else:
        print("  WARNING: Could not determine external IP.")

    print("VM started. MLflow service will be available shortly.")


def cmd_status() -> int:
    """Check VM and MLflow health. Returns 0 (ready), 1 (not ready), 2 (VM down)."""
    status = vm_status()
    if status is None:
        print(f"VM {MLFLOW_VM_NAME}: does not exist")
        return 2
    if status != "RUNNING":
        print(f"VM {MLFLOW_VM_NAME}: {status}")
        return 2

    ip = vm_external_ip()
    print(f"VM {MLFLOW_VM_NAME}: RUNNING (IP: {ip})")

    # Check MLflow process
    svc_result = ssh_to_vm(MLFLOW_VM_NAME, ZONE,
                           "systemctl is-active mlflow",
                           check=False)
    svc_status = svc_result.stdout.strip()
    if svc_status != "active":
        print(f"  MLflow service: {svc_status}")
        return 1

    # Check MLflow HTTP endpoint
    uri = f"http://{ip}:{MLFLOW_PORT}/health"
    import urllib.request
    import urllib.error
    try:
        resp = urllib.request.urlopen(uri, timeout=5)
        if resp.status == 200:
            print(f"  MLflow service: active, accepting connections")
            return 0
    except (urllib.error.URLError, OSError) as e:
        print(f"  MLflow service: active but not responding ({e})")
        return 1

    return 1


def cmd_stop():
    """Stop the MLflow VM (preserves disk)."""
    status = vm_status()
    if status is None:
        print(f"VM {MLFLOW_VM_NAME} does not exist.")
        return
    if status == "TERMINATED":
        print(f"VM {MLFLOW_VM_NAME} is already stopped.")
        return

    # Trigger a backup before stopping
    print("Backing up MLflow database...")
    ssh_to_vm(MLFLOW_VM_NAME, ZONE,
              f"gcloud storage cp ~/mlflow/mlflow.db {GCS_BACKUP_DIR}/mlflow.db",
              check=False)

    print(f"Stopping VM {MLFLOW_VM_NAME}...")
    gcloud("compute", "instances", "stop", MLFLOW_VM_NAME, f"--zone={ZONE}")
    print("VM stopped.")


def cmd_delete():
    """Delete the MLflow VM. Does NOT remove GCS backup."""
    status = vm_status()
    if status is None:
        print(f"VM {MLFLOW_VM_NAME} does not exist.")
        return

    # Backup before delete
    if status == "RUNNING":
        print("Backing up MLflow database...")
        ssh_to_vm(MLFLOW_VM_NAME, ZONE,
                  f"gcloud storage cp ~/mlflow/mlflow.db {GCS_BACKUP_DIR}/mlflow.db",
                  check=False)

    print(f"Deleting VM {MLFLOW_VM_NAME}...")
    gcloud("compute", "instances", "delete", MLFLOW_VM_NAME,
           f"--zone={ZONE}", "--quiet")
    print(f"VM deleted. Backup retained at {GCS_BACKUP_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "create":
        new_creds = "--new-credentials" in sys.argv
        cmd_create(new_credentials=new_creds)
    elif command == "start":
        cmd_start()
    elif command == "status":
        sys.exit(cmd_status())
    elif command == "stop":
        cmd_stop()
    elif command == "delete":
        cmd_delete()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
