#!/usr/bin/env python3
"""Run a vLLM parameter sweep and log results to MLflow.

This script is environment-agnostic: it reads MLflow configuration from
environment variables (MLFLOW_TRACKING_URI, MLFLOW_DEFAULT_ARTIFACT_ROOT)
and works identically in the local devcontainer and on a cloud VM.

Usage:
    python exp/vllm-sweeps/run-sweep.py
    python exp/vllm-sweeps/run-sweep.py --config exp/vllm-sweeps/sweep-config.yaml
"""

import argparse
import itertools
import os
from pathlib import Path

import mlflow
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_single(model: str, quantization: str, params: dict):
    """Run vLLM with a single parameter combination and log to MLflow."""
    with mlflow.start_run():
        # Log git metadata if available
        for key in ("BRANCH", "COMMIT", "RUN_ID"):
            val = os.environ.get(key)
            if val:
                mlflow.log_param(key.lower(), val)

        # Log sweep parameters
        mlflow.log_param("model", model)
        mlflow.log_param("quantization", quantization)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # TODO: Replace with actual vLLM benchmark invocation.
        # This placeholder logs the parameters and exits successfully,
        # which is enough to validate the infrastructure end-to-end.
        print(f"  model={model} quantization={quantization} {params}")
        mlflow.log_metric("placeholder", 1.0)


def main():
    default_config = Path(__file__).resolve().parent / "sweep-config.yaml"
    parser = argparse.ArgumentParser(description="Run vLLM parameter sweep.")
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to sweep config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model = config["model"]
    quantization = config["quantization"]
    sweep = config.get("sweep", {})

    # Build parameter grid from sweep config
    keys = list(sweep.keys())
    values = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in keys]
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    mlflow.set_experiment("vllm-sweeps")
    print(f"Running {len(grid)} parameter combinations...")

    for i, params in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}]", end="")
        run_single(model, quantization, params)

    print("Sweep complete.")


if __name__ == "__main__":
    main()
