#!/usr/bin/env bash
# Start vLLM with torch profiler enabled for TinyLlama.
# Traces land in TRACE_DIR as gzipped Chrome JSON.

set -euo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRACE_DIR="/tmp/vllm-traces"

mkdir -p "$TRACE_DIR"

vllm serve "$MODEL" \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir "$TRACE_DIR" \
  --profiler-config.torch_profiler_with_stack true \
  --profiler-config.torch_profiler_use_gzip true \
  --profiler-config.delay_iterations 2 \
  --profiler-config.max_iterations 5
