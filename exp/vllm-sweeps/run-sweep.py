#!/usr/bin/env python3
"""Run a vLLM parameter sweep and log results to MLflow.

This script is environment-agnostic: it reads MLflow configuration from
environment variables (MLFLOW_TRACKING_URI) or ~/.mlflow/server, and
works identically in the local devcontainer and on a cloud VM.

The parameter space is a banded grid: each (input_len, output_len) pair is
assigned to the smallest max_model_len that fits it. Iterate over vLLM server
parameters first to avoid excessive server restarts. ParameterSpace iterates
the space without materializing the grid.

Server params that cause OOM at startup trigger short-circuiting: all
remaining combos with strictly higher memory pressure are skipped.

Progress is tracked via an MLflow experiment tag ("n_completed"). The
deterministic iteration order means resume just skips the first N entries.

Usage:
    python exp/vllm-sweeps/run-sweep.py
    python exp/vllm-sweeps/run-sweep.py --name my-sweep
    python exp/vllm-sweeps/run-sweep.py --resume           # continue from last completed
    python exp/vllm-sweeps/run-sweep.py --resume 100       # continue from iteration 100
"""

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import yaml


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# MLflow connection
# ---------------------------------------------------------------------------

MLFLOW_SERVER_FILE = Path.home() / ".mlflow" / "server"


def resolve_tracking_uri() -> str:
    """Determine the MLflow tracking URI.

    Priority:
      1. MLFLOW_TRACKING_URI env var (already set)
      2. ~/.mlflow/server file
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    if MLFLOW_SERVER_FILE.exists():
        uri = MLFLOW_SERVER_FILE.read_text().strip()
        if uri:
            return uri
    sys.exit(
        "ERROR: No MLflow tracking URI found.\n"
        "Set MLFLOW_TRACKING_URI or create ~/.mlflow/server"
    )


def check_mlflow_health(uri: str):
    """Verify the MLflow server is reachable before starting the sweep.

    The /health endpoint is exempt from basic-auth in MLflow, so this
    works regardless of authentication configuration.
    """
    try:
        resp = urllib.request.urlopen(f"{uri}/health", timeout=10)
        if resp.status == 200:
            print(f"MLflow server: OK ({uri})")
            return
    except (urllib.error.URLError, OSError):
        pass
    sys.exit(f"ERROR: MLflow server not reachable at {uri}")


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

    SERVER_KEYS = {"quantization", "gpu_memory_utilization", "max_model_len",
                    "block_size"}

    def __init__(self, config: dict):
        server = config.get("server", {})
        workload = config.get("workload", {})
        self.checkpoints = config.get("checkpoints", {})
        self.model_base = config["model"]

        self.quantizations = sorted(server["quantization"])
        self.gpu_mem_utils = sorted(server["gpu_memory_utilization"])
        self.max_model_lens = sorted(server["max_model_len"])
        self.block_sizes = sorted(server.get("block_size", [None]))
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
            * len(self.block_sizes)
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

    @staticmethod
    def run_name(params: dict) -> str:
        """Generate a human-readable run name from params."""
        quant = params["quantization"]
        gpu = f"g{int(params['gpu_memory_utilization'] * 100)}"
        ml = f"ml{params['max_model_len']}"
        io = f"in{params['input_len']}-out{params['output_len']}"
        conc = f"c{params['concurrency']}"
        name = f"{quant}-{gpu}-{ml}-{io}-{conc}"
        if params.get("block_size") is not None:
            name += f"-bs{params['block_size']}"
        return name

    def __iter__(self):
        for quantization in self.quantizations:
            for gpu_mem in self.gpu_mem_utils:
                for block_size in self.block_sizes:
                    for band_index, max_len in enumerate(self.max_model_lens):
                        for input_len, output_len in self.band_pairs[band_index]:
                            for concurrency in self.concurrencies:
                                params = {
                                    "quantization": quantization,
                                    "gpu_memory_utilization": gpu_mem,
                                    "max_model_len": max_len,
                                    "input_len": input_len,
                                    "output_len": output_len,
                                    "concurrency": concurrency,
                                    "num_prompts": self.num_prompts,
                                    "num_warmups": self.num_warmups,
                                }
                                if block_size is not None:
                                    params["block_size"] = block_size
                                yield params


# Memory-pressure params: sorted so low pressure comes first, enabling
# short-circuit on OOM.  Higher gpu_memory_utilization = more pressure;
# higher max_model_len = more pressure.
MEMORY_PRESSURE_KEYS = ("gpu_memory_utilization", "max_model_len")


_OOM_PATTERNS = ("out of memory", "oom", "cuda error", "cublas error",
                  "not enough memory", "torch.cuda.outofmemoryerror")


def _looks_like_oom(msg: str) -> bool:
    """Heuristic: does the failure message suggest a GPU memory issue?"""
    lower = msg.lower()
    return any(p in lower for p in _OOM_PATTERNS)


def is_pareto_dominated(failed: dict, candidate: dict) -> bool:
    """True if candidate is strictly dominated: >= on all axes, > on at least one."""
    dominated = all(
        candidate.get(k, 0) >= failed.get(k, 0) for k in MEMORY_PRESSURE_KEYS
    )
    strictly = any(
        candidate.get(k, 0) > failed.get(k, 0) for k in MEMORY_PRESSURE_KEYS
    )
    return dominated and strictly


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

class VllmServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000,
                 startup_timeout: int = 300, log_dir: str | None = None):
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None
        self.base_url = f"http://{host}:{port}"
        self.log_dir = Path(log_dir) if log_dir else Path(tempfile.gettempdir())
        self.log_file: Path | None = None
        self._log_fh = None

    def start(self, model: str, server_params: dict) -> tuple[bool, str]:
        """Start vllm serve. Returns (healthy, diagnostic_message)."""
        cmd = ["vllm", "serve", model, "--host", self.host, "--port", str(self.port)]
        for key, value in server_params.items():
            flag = f"--{key.replace('_', '-')}"
            cmd.extend([flag, str(value)])

        print(f"  Starting vLLM server: {' '.join(cmd)}")
        self._close_log()
        self.log_file = self.log_dir / f"vllm-server-{int(time.time())}.log"
        self._log_fh = open(self.log_file, "w")
        self.process = subprocess.Popen(
            cmd, stdout=self._log_fh, stderr=subprocess.STDOUT,
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
        self._close_log()

    def _close_log(self):
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def is_healthy(self) -> bool:
        """Active health check via /health endpoint."""
        try:
            resp = urllib.request.urlopen(
                f"{self.base_url}/health", timeout=5)
            return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def drain_or_restart(self, model: str, server_params: dict,
                         drain_timeout: int = 30) -> bool:
        """Check server health after a timeout, restart if unresponsive.

        Returns True if the server is healthy afterward.
        """
        if not self.is_alive():
            print("  Server process is dead — restarting.")
            self.stop()
            healthy, msg = self.start(model, server_params)
            if not healthy:
                print(f"  Restart failed: {msg[:200]}")
            return healthy

        # Server process alive — check if it's actually responding
        deadline = time.monotonic() + drain_timeout
        while time.monotonic() < deadline:
            if self.is_healthy():
                print("  Server is healthy, continuing.")
                return True
            time.sleep(2)

        # Server alive but not responding
        print("  Server not responding — restarting.")
        logs = self._read_logs()
        if logs:
            print(f"  Server logs (last 1000 chars):\n{logs[-1000:]}")
        self.stop()
        healthy, msg = self.start(model, server_params)
        if not healthy:
            print(f"  Restart failed: {msg[:200]}")
        return healthy

    def _read_logs(self) -> str:
        if self.log_file and self.log_file.exists():
            try:
                return self.log_file.read_text(errors="replace")[-4000:]
            except Exception:
                pass
        return ""


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

_TIMEOUT = "TIMEOUT"


def run_benchmark(base_url: str, model: str, workload: dict,
                  bench_config: dict, result_dir: Path) -> dict | str | None:
    """Run `vllm bench serve` and return parsed JSON, _TIMEOUT, or None."""
    timeout = int(bench_config.get("benchmark_timeout", 1800))
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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        print(f"  Benchmark timed out ({timeout}s)")
        bench_output = ""
        if exc.stdout:
            bench_output += exc.stdout[-2000:] if isinstance(exc.stdout, str) \
                else exc.stdout.decode(errors="replace")[-2000:]
        if exc.stderr:
            bench_output += exc.stderr[-2000:] if isinstance(exc.stderr, str) \
                else exc.stderr.decode(errors="replace")[-2000:]
        if bench_output:
            print(f"  Benchmark output:\n{bench_output}")
        return _TIMEOUT

    if result.returncode != 0:
        print(f"  Benchmark failed (rc={result.returncode}): {result.stderr[-500:]}")
        return None

    json_files = sorted(result_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not json_files:
        print("  Benchmark produced no JSON output")
        return None

    return json.loads(json_files[-1].read_text())


# Aggregate throughput and request-count metrics from vllm bench serve output.
SUMMARY_METRIC_KEYS = (
    "duration", "completed", "failed",
    "request_throughput", "output_throughput", "total_token_throughput",
    "max_output_tokens_per_s", "max_concurrent_requests",
)

# Per-metric latency statistics for TTFT, TPOT, ITL, and E2EL.
# Percentile keys assume metric_percentiles="50,90,95,99" in the config.
LATENCY_METRIC_KEYS = (
    "mean_ttft_ms", "median_ttft_ms", "std_ttft_ms",
    "p50_ttft_ms", "p90_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms",
    "p50_tpot_ms", "p90_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "std_itl_ms",
    "p50_itl_ms", "p90_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "mean_e2el_ms", "median_e2el_ms", "std_e2el_ms",
    "p50_e2el_ms", "p90_e2el_ms", "p95_e2el_ms", "p99_e2el_ms",
)


def extract_metrics(result_json: dict) -> dict[str, float]:
    """Pull benchmark metrics from vllm bench serve JSON output."""
    metrics = {}
    for key in SUMMARY_METRIC_KEYS + LATENCY_METRIC_KEYS:
        if key in result_json:
            metrics[key] = float(result_json[key])
    return metrics


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_params(model: str, server_params: dict, workload_params: dict):
    """Log sweep cell parameters to the active MLflow run."""
    for env_key in ("BRANCH", "COMMIT"):
        val = os.environ.get(env_key)
        if val:
            mlflow.log_param(env_key.lower(), val)

    mlflow.log_param("model", model)
    for k, v in server_params.items():
        mlflow.log_param(k, v)
    for k, v in workload_params.items():
        mlflow.log_param(k, v)


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

def main():
    default_config = Path(__file__).resolve().parent / "sweep-config.yaml"
    parser = argparse.ArgumentParser(description="Run vLLM parameter sweep.")
    parser.add_argument("--config", default=str(default_config),
                        help="Path to sweep config YAML")
    parser.add_argument("--name", default=None,
                        help="Experiment name. Auto-generated as "
                             "vllm-sweep-MMDD-HHMM if not specified. "
                             "Required when using --resume.")
    parser.add_argument("--resume", nargs="?", const=-1, type=int, default=None,
                        help="Resume a sweep. Optionally specify iteration index "
                             "to start from. Without a value, resumes from the "
                             "last completed iteration. Requires --name.")
    args = parser.parse_args()

    if args.resume is not None and args.name is None:
        parser.error("--resume requires --name to identify the experiment")

    config = load_config(args.config)
    bench_config = config.get("bench", {})
    serve_config = config.get("serve", {})

    # Discover and verify MLflow tracking server
    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    check_mlflow_health(tracking_uri)

    sweep = ParameterSpace(config)
    total = len(sweep)

    # Experiment naming: use provided name, or generate MMDD-HHMM
    if args.name:
        experiment_name = args.name
    else:
        from datetime import datetime
        experiment_name = f"vllm-sweep-{datetime.now().strftime('%m%d-%H%M')}"

    client = MlflowClient()
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id

    # Determine starting point
    start_from = 0
    if args.resume is not None:
        if args.resume >= 0:
            start_from = args.resume
        else:
            exp = client.get_experiment(experiment_id)
            completed_str = exp.tags.get("n_completed", "0")
            start_from = int(completed_str)

        if start_from >= total:
            print(f"Sweep already complete ({start_from}/{total}).")
            return

        if start_from > 0:
            print(f"Resuming from {start_from}/{total}")

    print(f"Experiment: {experiment_name}")
    print(f"Sweep: {total} total runs, starting at {start_from}")

    server = VllmServer(
        host=serve_config.get("host", "127.0.0.1"),
        port=serve_config.get("port", 8000),
        startup_timeout=serve_config.get("startup_timeout", 300),
    )

    oom_combos: list[dict] = []
    prev_server_config: dict | None = None

    for run_idx, params in itertools.islice(enumerate(sweep), start_from, None):
        server_params = sweep.server_config(params)
        workload_params = sweep.workload_config(params)
        model_path = sweep.model_path(params)
        name = ParameterSpace.run_name(params)

        try:
            with mlflow.start_run(run_name=name):
                log_params(model_path, server_params, workload_params)

                # Short-circuit: skip if higher pressure than a prior OOM
                if any(is_pareto_dominated(oom, server_params)
                       for oom in oom_combos):
                    mlflow.log_param("error",
                                     "Skipped: Pareto-dominated by prior OOM config")
                    mlflow.end_run("FAILED")
                    continue

                # Restart server if config changed
                if server_params != prev_server_config:
                    server.stop()
                    print(f"\n[server] {server_params}")
                    healthy, msg = server.start(model_path, server_params)
                    if not healthy:
                        print(f"  Server startup failed: {msg[:200]}")
                        if _looks_like_oom(msg):
                            oom_combos.append(server_params)
                        mlflow.log_param("error",
                                         f"Server startup failed: {msg[:200]}")
                        mlflow.end_run("FAILED")
                        prev_server_config = server_params
                        continue
                    prev_server_config = server_params

                # Check server health before each benchmark
                if not server.is_alive():
                    print("  Server crashed — attempting restart")
                    mlflow.log_param("error", "Server crashed during benchmark")
                    mlflow.end_run("FAILED")
                    server.stop()
                    healthy, msg = server.start(model_path, server_params)
                    if not healthy and _looks_like_oom(msg):
                        oom_combos.append(server_params)
                    prev_server_config = server_params
                    continue

                with tempfile.TemporaryDirectory(prefix="bench_") as tmpdir:
                    result = run_benchmark(
                        server.base_url, model_path, workload_params,
                        bench_config, Path(tmpdir),
                    )

                if result is _TIMEOUT:
                    mlflow.log_param("error", "Benchmark timed out")
                    if server.log_file and server.log_file.exists():
                        mlflow.log_artifact(str(server.log_file),
                                            "diagnostics")
                    mlflow.end_run("FAILED")
                    server.drain_or_restart(model_path, server_params)
                elif result is None:
                    mlflow.log_param("error", "Benchmark client failed")
                    mlflow.end_run("FAILED")
                else:
                    metrics = extract_metrics(result)
                    for k, v in metrics.items():
                        mlflow.log_metric(k, v)
                    print(f"  → throughput={metrics.get('request_throughput', '?'):.2f} req/s "
                          f"ttft={metrics.get('mean_ttft_ms', '?'):.1f}ms")
                    # context manager exits → FINISHED
        finally:
            client.set_experiment_tag(experiment_id, "n_completed",
                                      str(run_idx + 1))

    server.stop()
    print("Sweep complete.")


if __name__ == "__main__":
    main()
