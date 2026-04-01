#!/usr/bin/env bash
# Trigger profiling on a running vLLM server, fire requests, then stop.
# Run profile-serve.sh first in another terminal.

set -euo pipefail

BASE_URL="http://localhost:8000"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
N_REQUESTS=10

# Warmup — one request to get past any lazy init
echo "Sending warmup request..."
curl -s "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"Hello world\", \"max_tokens\": 16}" \
  > /dev/null

# Start profiling
echo "Starting profiler..."
curl -s -X POST "$BASE_URL/start_profile"
echo

# Fire requests
echo "Sending $N_REQUESTS requests..."
for i in $(seq 1 "$N_REQUESTS"); do
  curl -s "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Explain how transformer attention works\", \"max_tokens\": 128}" \
    > /dev/null &
done
wait

# Stop profiling
echo "Stopping profiler..."
curl -s -X POST "$BASE_URL/stop_profile"
echo

echo "Done. Traces should be in /tmp/vllm-traces/"
echo "Run: python -m xprofiler summary /tmp/vllm-traces/*.json.gz --model llama"
