#!/usr/bin/env python3
"""Run a vLLM parameter sweep and log results to MLflow.

This script is environment-agnostic: it reads MLflow configuration from
environment variables (MLFLOW_TRACKING_URI, MLFLOW_DEFAULT_ARTIFACT_ROOT)
and works identically in the local devcontainer and on a cloud VM.

The sweep iterates over server configs (sorted by increasing memory pressure)
and, within each, over valid workload combos (banded by max_model_len so that
input_len + output_len < max_model_len).  Workload combos that fit under a
smaller max_model_len are assigned there, not duplicated at larger values.

Server params that cause OOM at startup trigger short-circuiting: all
remaining combos with strictly higher memory pressure are skipped.

Progress is saved as (server_idx, workload_idx) so the sweep can be
interrupted and resumed from where it left off.

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


# ---------------------------------------------------------------------------
# Config parsing & grid construction
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_grid(section: dict) -> list[dict]:
    """Expand a config section into a list of param dicts (cartesian product).

    Scalar values are treated as single-element lists.
    """
    keys = list(section.keys())
    values = [v if isinstance(v, list) else [v] for v in section.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def build_banded_schedule(server_combos: list[dict],
                          workload_combos: list[dict],
                          ) -> list[tuple[dict, list[dict]]]:
    """Assign workload combos to server configs via max_model_len banding.

    Each (input_len, output_len) pair is assigned to the smallest
    max_model_len that can accommodate it.  Workloads are not duplicated
    across bands.

    Returns a list of (server_params, workloads) pairs, sorted by increasing
    memory pressure within server configs.
    """
    # Collect sorted max_model_len values from the server combos
    max_lens = sorted({c["max_model_len"] for c in server_combos})

    # Assign each (input_len, output_len) pair to its smallest valid band
    pair_to_band: dict[tuple[int, int], int] = {}
    for w in workload_combos:
        pair = (w["input_len"], w["output_len"])
        if pair in pair_to_band:
            continue
        total = pair[0] + pair[1]
        for ml in max_lens:
            if total < ml:
                pair_to_band[pair] = ml
                break
        # If no band fits, the pair is invalid and will be omitted

    # Build per-server-config workload lists
    schedule = []
    for sc in server_combos:
        ml = sc["max_model_len"]
        valid_workloads = [
            w for w in workload_combos
            if pair_to_band.get((w["input_len"], w["output_len"])) == ml
        ]
        schedule.append((sc, valid_workloads))

    return schedule


# Memory-pressure params: sorted so low pressure comes first, enabling
# short-circuit on OOM.  Higher gpu_memory_utilization = more pressure;
# higher max_model_len = more pressure.
MEMORY_PRESSURE_KEYS = ("gpu_memory_utilization", "max_model_len")


def memory_pressure_sort_key(combo: dict) -> tuple:
    """Sort key: increasing memory pressure."""
    return tuple(combo.get(k, 0) for k in MEMORY_PRESSURE_KEYS)


def is_strictly_higher_pressure(failed: dict, candidate: dict) -> bool:
    """True if candidate has >= pressure on ALL memory axes vs failed."""
    return all(
        candidate.get(k, 0) >= failed.get(k, 0) for k in MEMORY_PRESSURE_KEYS
    )


# ---------------------------------------------------------------------------
# Progress / pause-resume
# ---------------------------------------------------------------------------

def load_progress(path: str | None) -> tuple[int, int]:
    """Load (server_idx, workload_idx) from progress file, or (0, 0)."""
    if path is None:
        return 0, 0
    p = Path(path)
    if not p.exists():
        return 0, 0
    try:
        data = json.loads(p.read_text())
        return data.get("server_idx", 0), data.get("workload_idx", 0)
    except (json.JSONDecodeError, OSError):
        return 0, 0


def save_progress(path: str | None, server_idx: int, workload_idx: int,
                  total: int):
    """Atomically write progress state."""
    if path is None:
        return
    data = {"server_idx": server_idx, "workload_idx": workload_idx,
            "total": total}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.write(fd, json.dumps(data).encode())
        os.fsync(fd)
    finally:
        os.close(fd)


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
    model_base = config["model"]
    checkpoints = config.get("checkpoints", {})
    server_section = config.get("server", {})
    workload_section = config.get("workload", {})
    bench_config = config.get("bench", {})
    serve_config = config.get("serve", {})

    server_combos = expand_grid(server_section)
    workload_combos = expand_grid(workload_section)

    # Sort server combos by increasing memory pressure for OOM short-circuiting
    server_combos.sort(key=memory_pressure_sort_key)

    # Build banded schedule: each server config gets only the workload combos
    # whose (input_len + output_len) fits under its max_model_len band
    schedule = build_banded_schedule(server_combos, workload_combos)

    total_runs = sum(len(wl) for _, wl in schedule)

    mlflow.set_experiment("vllm-sweeps")

    # Resume support: skip past already-completed work
    resume_server_idx, resume_workload_idx = load_progress(args.progress)
    if resume_server_idx > 0 or resume_workload_idx > 0:
        print(f"Resuming from server_idx={resume_server_idx}, "
              f"workload_idx={resume_workload_idx}")

    print(f"Sweep: {len(schedule)} server configs, {total_runs} total runs")

    server = VllmServer(
        host=serve_config.get("host", "127.0.0.1"),
        port=serve_config.get("port", 8000),
        startup_timeout=serve_config.get("startup_timeout", 300),
    )

    # Track OOM'd server combos for short-circuiting
    oom_combos: list[dict] = []

    for si, (server_params, valid_workloads) in enumerate(schedule):
        if si < resume_server_idx:
            continue

        quant = server_params.get("quantization", "")
        model_path = checkpoints.get(quant, model_base)

        if not valid_workloads:
            continue

        # Short-circuit: skip if any prior OOM has strictly lower pressure
        if any(is_strictly_higher_pressure(oom, server_params)
               for oom in oom_combos):
            print(f"\n[skip] Server config {server_params} — "
                  f"higher pressure than a prior OOM")
            for wi, wp in enumerate(valid_workloads):
                log_run(model_path, server_params, wp,
                        error="Skipped: higher memory pressure than prior OOM")
                save_progress(args.progress, si, wi + 1, total_runs)
            continue

        # Server CLI params: quantization is passed explicitly for kernel
        # selection; it is not filtered out.
        serve_params = {k: v for k, v in server_params.items()}

        print(f"\n[server {si + 1}/{len(schedule)}] {server_params} "
              f"({len(valid_workloads)} workloads)")

        server.stop()
        healthy, msg = server.start(model_path, serve_params)

        if not healthy:
            print(f"  Server startup failed: {msg[:200]}")
            oom_combos.append(server_params)
            for wi, wp in enumerate(valid_workloads):
                if si == resume_server_idx and wi < resume_workload_idx:
                    continue
                log_run(model_path, server_params, wp,
                        error=f"Server startup failed: {msg[:200]}")
                save_progress(args.progress, si, wi + 1, total_runs)
            continue

        for wi, workload_params in enumerate(valid_workloads):
            if si == resume_server_idx and wi < resume_workload_idx:
                continue

            if not server.is_alive():
                print("  Server crashed — attempting restart")
                log_run(model_path, server_params, workload_params,
                        error="Server crashed during benchmark")
                save_progress(args.progress, si, wi + 1, total_runs)

                server.stop()
                healthy, msg = server.start(model_path, serve_params)
                if not healthy:
                    print(f"  Restart failed: {msg[:200]}")
                    for wp in valid_workloads[wi + 1:]:
                        log_run(model_path, server_params, wp,
                                error=f"Server restart failed: {msg[:200]}")
                    save_progress(args.progress, si, len(valid_workloads),
                                  total_runs)
                    break
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

            save_progress(args.progress, si, wi + 1, total_runs)

    server.stop()
    print("Sweep complete.")


if __name__ == "__main__":
    main()
