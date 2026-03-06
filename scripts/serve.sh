#!/usr/bin/env bash
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Add it to ~/.hftoken on the host and rebuild the container."
    exit 1
fi

MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

exec vllm serve "$MODEL" \
    --quantization awq_marlin \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000
