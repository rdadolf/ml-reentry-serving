#!/usr/bin/env python3
"""Run a vLLM parameter sweep and log results to MLflow.

This script is environment-agnostic: it reads MLflow configuration from
environment variables (MLFLOW_TRACKING_URI, MLFLOW_DEFAULT_ARTIFACT_ROOT)
and works identically in the local devcontainer and on a cloud VM.

The parameter space is a banded grid: each (input_len, output_len) pair is
assigned to the smallest max_model_len that fits it. Iterate over vLLM server
parameters first to avoid excessive server restarts. ParameterSpace iterates
the space without materializing the grid, and supports index-based lookup
for progress tracking and resume.

Server params that cause OOM at startup trigger short-circuiting: all
remaining combos with strictly higher memory pressure are skipped.

Progress is tracked as a scalar completed count via ExperimentProgress.
The deterministic iteration order means resume just skips the first N entries.

Usage:
    python exp/vllm-sweeps/run-sweep.py
    python exp/vllm-sweeps/run-sweep.py --config exp/vllm-sweeps/sweep-config.yaml
"""

import argparse
import itertools
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

import mlflow
import yaml

from experiment_progress import ExperimentProgress


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

class ParameterSpace:
    """Banded parameter space with deterministic iteration order.

    Iteration order (outer to inner):
        quantization → gpu_memory_utilization → max_model_len →
        (input_len, output_len) band pairs → concurrency

    Sorted by increasing memory pressure so OOM short-circuiting is effective.
    Each (input_len, output_len) pair is assigned to the smallest
    max_model_len band where input_len + output_len < max_model_len.

    All swept parameters must be lists in the config.
    """

    SERVER_KEYS = {"quantization", "gpu_memory_utilization", "max_model_len"}

    def __init__(self, config: dict):
        server = config.get("server", {})
        workload = config.get("workload", {})
        self.checkpoints = config.get("checkpoints", {})
        self.model_base = config["model"]

        self.quantizations = sorted(server["quantization"])
        self.gpu_mem_utils = sorted(server["gpu_memory_utilization"])
        self.max_model_lens = sorted(server["max_model_len"])
        self.concurrencies = sorted(workload["concurrency"])

        input_lens = sorted(workload["input_len"])
        output_lens = sorted(workload["output_len"])

        self.num_prompts = workload.get("num_prompts", 100)
        self.num_warmups = workload.get("num_warmups", 5)

        # Band assignment: each (input_len, output_len) pair goes to the
        # smallest max_model_len where input_len + output_len < max_model_len.
        self.band_pairs: list[list[tuple[int, int]]] = [
            [] for _ in self.max_model_lens
        ]
        for input_len in input_lens:
            for output_len in output_lens:
                for band_index, max_len in enumerate(self.max_model_lens):
                    if input_len + output_len < max_len:
                        self.band_pairs[band_index].append(
                            (input_len, output_len))
                        break

        self.workloads_per_band = [
            len(pairs) * len(self.concurrencies)
            for pairs in self.band_pairs
        ]
        self.workloads_per_gpu_mem = sum(self.workloads_per_band)
        self._total = (
            len(self.quantizations)
            * len(self.gpu_mem_utils)
            * self.workloads_per_gpu_mem
        )

    def __len__(self) -> int:
        return self._total

    def server_config(self, params: dict) -> dict:
        """Extract server-side params (those requiring a restart)."""
        return {k: v for k, v in params.items() if k in self.SERVER_KEYS}

    def workload_config(self, params: dict) -> dict:
        """Extract workload-side params (those that don't need restart)."""
        return {k: v for k, v in params.items() if k not in self.SERVER_KEYS}

    def model_path(self, params: dict) -> str:
        """Resolve quantization to HF checkpoint path."""
        return self.checkpoints.get(params["quantization"], self.model_base)

    def __iter__(self):
        for quantization in self.quantizations:
            for gpu_mem in self.gpu_mem_utils:
                for band_index, max_len in enumerate(self.max_model_lens):
                    for input_len, output_len in self.band_pairs[band_index]:
                        for concurrency in self.concurrencies:
                            yield {
                                "quantization": quantization,
                                "gpu_memory_utilization": gpu_mem,
                                "max_model_len": max_len,
                                "input_len": input_len,
                                "output_len": output_len,
                                "concurrency": concurrency,
                                "num_prompts": self.num_prompts,
                                "num_warmups": self.num_warmups,
                            }


# Memory-pressure params: sorted so low pressure comes first, enabling
# short-circuit on OOM.  Higher gpu_memory_utilization = more pressure;
# higher max_model_len = more pressure.
MEMORY_PRESSURE_KEYS = ("gpu_memory_utilization", "max_model_len")


def is_strictly_higher_pressure(failed: dict, candidate: dict) -> bool:
    """True if candidate has >= pressure on ALL memory axes vs failed."""
    return all(
        candidate.get(k, 0) >= failed.get(k, 0) for k in MEMORY_PRESSURE_KEYS
    )


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

class VllmServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000,
                 startup_timeout: int = 300):
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None
        self.base_url = f"http://{host}:{port}"

    def start(self, model: str, server_params: dict) -> tuple[bool, str]:
        """Start vllm serve. Returns (healthy, diagnostic_message)."""
        cmd = ["vllm", "serve", model, "--host", self.host, "--port", str(self.port)]
        for key, value in server_params.items():
            flag = f"--{key.replace('_', '-')}"
            cmd.extend([flag, str(value)])

        print(f"  Starting vLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return self._wait_healthy()

    def _wait_healthy(self) -> tuple[bool, str]:
        """Poll /health until ready, process death, or timeout.

        Returns (success, message).
        Three outcomes:
          1. Process died → startup failure (OOM or bad config)
          2. /health 200  → ready
          3. Timeout       → hang, kill and report
        """
        deadline = time.monotonic() + self.startup_timeout
        health_url = f"{self.base_url}/health"

        while time.monotonic() < deadline:
            rc = self.process.poll()
            if rc is not None:
                logs = self._read_logs()
                return False, f"Server exited with code {rc}.\n{logs}"

            try:
                resp = urllib.request.urlopen(health_url, timeout=2)
                if resp.status == 200:
                    return True, "healthy"
            except (urllib.error.URLError, OSError):
                pass

            time.sleep(2)

        # Timeout — kill the hung process
        self.stop()
        return False, f"Server did not become healthy within {self.startup_timeout}s"

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _read_logs(self) -> str:
        if self.process and self.process.stdout:
            try:
                return self.process.stdout.read().decode(errors="replace")[-4000:]
            except Exception:
                pass
        return ""


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(base_url: str, model: str, workload: dict,
                  bench_config: dict, result_dir: Path) -> dict | None:
    """Run `vllm bench serve` and return parsed JSON, or None on failure."""
    cmd = [
        "vllm", "bench", "serve",
        "--base-url", base_url,
        "--model", model,
        "--backend", str(bench_config.get("backend", "openai")),
        "--dataset-name", "random",
        "--random-input-len", str(workload["input_len"]),
        "--random-output-len", str(workload["output_len"]),
        "--max-concurrency", str(workload["concurrency"]),
        "--request-rate", str(bench_config.get("request_rate", "inf")),
        "--num-prompts", str(workload.get("num_prompts", 100)),
        "--num-warmups", str(workload.get("num_warmups", 5)),
        "--metric-percentiles", str(bench_config.get("metric_percentiles", "99")),
        "--save-result",
        "--save-detailed",
        "--result-dir", str(result_dir),
    ]

    print(f"  Benchmark: concurrency={workload['concurrency']} "
          f"input_len={workload['input_len']} output_len={workload['output_len']}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  Benchmark failed (rc={result.returncode}): {result.stderr[-500:]}")
        return None

    json_files = sorted(result_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not json_files:
        print("  Benchmark produced no JSON output")
        return None

    return json.loads(json_files[-1].read_text())


# Aggregate metrics to extract from vllm bench serve JSON output.
METRIC_KEYS = (
    "duration", "completed", "failed",
    "request_throughput", "output_throughput", "total_token_throughput",
    "max_output_tokens_per_s", "max_concurrent_requests",
)


def extract_metrics(result_json: dict) -> dict[str, float]:
    """Pull benchmark metrics from vllm bench serve JSON output."""
    metrics = {}
    for key in METRIC_KEYS:
        if key in result_json:
            metrics[key] = float(result_json[key])

    # Per-metric aggregates: mean/median/std/percentiles for ttft, tpot, itl, e2el
    for key, val in result_json.items():
        if key.startswith(("mean_", "median_", "std_", "p")) and key.endswith("_ms"):
            try:
                metrics[key] = float(val)
            except (ValueError, TypeError):
                pass

    return metrics


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_run(model: str, server_params: dict, workload_params: dict,
            metrics: dict | None = None, error: str | None = None):
    """Log one sweep cell to MLflow. Params are flat (no prefix)."""
    with mlflow.start_run():
        for env_key in ("BRANCH", "COMMIT", "RUN_ID"):
            val = os.environ.get(env_key)
            if val:
                mlflow.log_param(env_key.lower(), val)

        mlflow.log_param("model", model)
        for k, v in server_params.items():
            mlflow.log_param(k, v)
        for k, v in workload_params.items():
            mlflow.log_param(k, v)

        if error:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(error)[:250])
        else:
            mlflow.log_param("status", "success")
            for k, v in (metrics or {}).items():
                mlflow.log_metric(k, v)


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

def main():
    default_config = Path(__file__).resolve().parent / "sweep-config.yaml"
    parser = argparse.ArgumentParser(description="Run vLLM parameter sweep.")
    parser.add_argument("--config", default=str(default_config),
                        help="Path to sweep config YAML")
    parser.add_argument("--progress", default=None,
                        help="Path to write progress JSON (for monitoring "
                             "and pause/resume)")
    args = parser.parse_args()

    config = load_config(args.config)
    bench_config = config.get("bench", {})
    serve_config = config.get("serve", {})

    sweep = ParameterSpace(config)
    total = len(sweep)

    mlflow.set_experiment("vllm-sweeps")

    completed = 0
    if args.progress:
        completed = ExperimentProgress.load_completed(args.progress)
    if completed > 0:
        print(f"Resuming from {completed}/{total}")
    ExperimentProgress.init(total, args.progress)
    ExperimentProgress.set_completed(completed)

    print(f"Sweep: {total} total runs")

    server = VllmServer(
        host=serve_config.get("host", "127.0.0.1"),
        port=serve_config.get("port", 8000),
        startup_timeout=serve_config.get("startup_timeout", 300),
    )

    oom_combos: list[dict] = []
    prev_server_config: dict | None = None

    for params in itertools.islice(sweep, completed, None):
        server_params = sweep.server_config(params)
        workload_params = sweep.workload_config(params)
        model_path = sweep.model_path(params)

        # Short-circuit: skip if higher pressure than a prior OOM
        if any(is_strictly_higher_pressure(oom, server_params)
               for oom in oom_combos):
            log_run(model_path, server_params, workload_params,
                    error="Skipped: higher memory pressure than prior OOM")
            ExperimentProgress.step()
            continue

        # Restart server if config changed
        if server_params != prev_server_config:
            server.stop()
            print(f"\n[server] {server_params}")
            healthy, msg = server.start(model_path, server_params)
            if not healthy:
                print(f"  Server startup failed: {msg[:200]}")
                oom_combos.append(server_params)
                log_run(model_path, server_params, workload_params,
                        error=f"Server startup failed: {msg[:200]}")
                ExperimentProgress.step()
                prev_server_config = server_params
                continue
            prev_server_config = server_params

        # Check server health before each benchmark
        if not server.is_alive():
            print("  Server crashed — attempting restart")
            log_run(model_path, server_params, workload_params,
                    error="Server crashed during benchmark")
            ExperimentProgress.step()
            server.stop()
            healthy, msg = server.start(model_path, server_params)
            if not healthy:
                oom_combos.append(server_params)
            prev_server_config = server_params
            continue

        with tempfile.TemporaryDirectory(prefix="bench_") as tmpdir:
            result = run_benchmark(
                server.base_url, model_path, workload_params,
                bench_config, Path(tmpdir),
            )

        if result is None:
            log_run(model_path, server_params, workload_params,
                    error="Benchmark client failed")
        else:
            metrics = extract_metrics(result)
            log_run(model_path, server_params, workload_params,
                    metrics=metrics)
            print(f"  → throughput={metrics.get('request_throughput', '?'):.2f} req/s "
                  f"ttft={metrics.get('mean_ttft_ms', '?'):.1f}ms")

        ExperimentProgress.step()

    server.stop()
    print("Sweep complete.")


if __name__ == "__main__":
    main()
