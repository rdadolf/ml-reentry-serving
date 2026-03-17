# Benchmark Parameter Space

Model: `meta-llama/Meta-Llama-3.1-8B-Instruct` (GQA, 8 KV heads — 4x smaller KV cache than full MHA)

## Hardware Scenarios

| | GTX 4070 (personal) | A100 40GB (cloud) |
|---|---|---|
| VRAM | 12 GB | 40 GB |
| Memory bandwidth | ~504 GB/s (GDDR6X) | ~2.0 TB/s (HBM2e) |
| Estimated cost | Free | ~$1.50/hour |
| FP16 viable | No (weights alone ~14-16 GB) | Yes (tight — ~22 GB KV headroom) |

## Sweep Dimensions

### Quantization

| Config | Weights | GTX 4070 KV headroom | A100 40GB KV headroom |
|--------|---------|----------------------|-----------------------|
| INT4 (AWQ) | ~3.5-4 GB | ~7-8 GB | ~34 GB |
| INT8 | ~7-8 GB | ~4 GB | ~30 GB |
| FP16 (BF16) | ~14-16 GB | Does not fit | ~22 GB |

FP16 is excluded on the GTX 4070 — it doesn't fit, not a gap. On the A100 40GB, FP16 fits but with constrained KV headroom (~22 GB) — memory pressure becomes visible at high concurrency with long sequences.

### Concurrency (simultaneous requests)

| GTX 4070 | A100 40GB |
|----------|-----------|
| 1, 2, 4, 8, 16 | 1, 2, 4, 8, 16, 32, 64, 128 |

vLLM handles concurrency via continuous batching — there is no batch size flag. Concurrency is controlled by the number of simultaneous clients.

On the GTX 4070, expect OOM in the 8-16 range at INT8 with longer sequences. On the A100 40GB, INT4/INT8 have enough headroom for the full concurrency range. FP16 is tighter — expect preemption or OOM at C=64+ with longer sequences.

### Input Sequence Length (tokens)

| GTX 4070 | A100 40GB |
|----------|-----------|
| 128, 512, 1024, 2048 | 128, 512, 1024, 2048, 4096 |

Upper bound on the GTX 4070 is constrained by KV cache budget — INT4 may tolerate up to 4096, INT8 will be tighter. On the A100 40GB, INT4/INT8 can push to 4096. FP16 at high concurrency may need to cap at 2048.

### Output Sequence Length (tokens generated)

| GTX 4070 | A100 40GB |
|----------|-----------|
| 64, 128, 256, 512 | 64, 128, 256, 512, 1024 |

Constraint: input + output must stay within `max-model-len` (currently 4096).

## Sweep Design

Full grids are impractical (160 combos on GTX 4070, 600 on A100). Reduced designs:

### GTX 4070 (~50-70 runs)

- **Primary INT4 sweep:** All concurrency x sequence length combos (~80 nominal, minus OOM exclusions)
- **Targeted INT8 sweep:** Lower concurrency / shorter sequences where it fits (~20-30 runs)
- **Boundary probing:** Runs near predicted OOM to characterize failure mode (graceful preemption vs. crash)

### A100 40GB (~100-120 runs)

- **FP16 baseline sweep:** Concurrency up to 32 (64 at shorter sequences) x sequence lengths (~25-30 runs)
- **INT8 and INT4 comparison:** Match the FP16 sweep points plus higher concurrency/longer sequences (~60 runs)
- **Saturation probing:** High concurrency (64, 128) x long sequences at INT4/INT8; FP16 boundary probing at C=32-64 (~20 runs)

## Time and Cost Estimates

Based on measured throughput: 30 tok/s single-request decode, ~50 tok/s batched ceiling (GTX 4070, INT4). A100 numbers are extrapolated from hardware specs. Confidence: +/-50%.

### Throughput Assumptions

| Parameter | GTX 4070 | A100 40GB (estimated) |
|---|---|---|
| Decode tok/s (single request) | 30 (measured) | ~120 |
| Decode tok/s ceiling (batched) | 50 (measured) | ~2000+ |
| Prefill throughput | ~2000 tok/s | ~10,000+ tok/s |
| INT8 penalty vs INT4 | ~20% | ~10-15% |
| Per-run overhead (warmup, settle) | ~10-15s | ~10-15s |
| Server restart (quant change) | ~45-60s (1 restart) | ~30-45s (2 restarts) |

Throughput is the same as the 80GB variant — same memory bandwidth and compute. The 40GB card doesn't run slower per-token, it just runs out of KV cache sooner.

### Wall Clock Estimates

| | GTX 4070 | A100 40GB |
|---|---|---|
| Single iteration | ~1 hour | ~45 min - 1 hour |
| With 3 iterations per config | ~2.5-3 hours | ~2-3 hours |
| Cloud cost | Free | ~$3-5 |

The A100 is faster despite more runs because its batched throughput is much higher. The GTX 4070 sweep's long pole is the handful of high-concurrency, long-output INT4 runs. The A100 sweep's long pole is saturation probing at C=128 (INT4/INT8). FP16 runs that OOM fail fast and don't add significant time.

## Metrics

- **Throughput:** tokens/second (prefill and generation separately)
- **Latency:** time-to-first-token (TTFT), inter-token latency, end-to-end request latency
- **GPU utilization:** VRAM usage over time, compute utilization
- **Failure modes:** OOM boundaries, request preemption/requeuing by the scheduler

## Analysis Questions

- Where does throughput saturate as concurrency increases? (Scheduler batching limit)
- How does TTFT degrade under load? (Queuing and preemption behavior)
- Where does the INT4 vs INT8 (vs FP16 on A100 40GB) throughput/latency frontier diverge? (Compute-vs-memory tradeoff)
- At what (concurrency x sequence length) product does request preemption begin? (PagedAttention memory management)
- How does vLLM behave approaching OOM — graceful degradation or cliff?

## Comparison Summary

| | GTX 4070 | A100 40GB |
|---|---|---|
| Quant levels | INT4, INT8 | INT4, INT8, FP16 |
| Concurrency range | 1-16 | 1-128 (FP16: up to 32-64) |
| Max practical input | 2048 | 4096 (FP16 at high C: 2048) |
| Output range | 64-512 | 64-1024 |
| OOM boundary | Central to the analysis | Visible at FP16/high-C frontier |
| Primary story | Memory management under pressure | Throughput scaling + memory pressure at FP16 boundary |
| Unique strength | Scheduler decisions visible because memory is tight | Full quant frontier with memory pressure still observable at FP16 |
| Estimated cost | Free | ~$3-5 |
