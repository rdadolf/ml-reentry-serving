# Memory footprint estimates

(Claude analysis, using `vllm-sweep-0317-2153` data)

The vLLM server logs from our g85/ml1024 run gave us measured anchor points. The estimates below are built from those.

## Model weights (measured)

From the server log: "Model loading took **5.37 GiB** memory." This is the AWQ INT4 model — 8B params at 4 bits = 4 GiB for quantized layers, plus ~1.4 GiB for non-quantized layers (embeddings, LM head at FP16: 128,256 vocab × 4,096 hidden × 2 bytes × 2 ≈ 2 GiB, minus weight tying if any).

## CUDA graphs (measured)

From server log: "Graph capturing... took **0.57 GiB**"

## KV cache (derived from measured)

From server log at g85:
- "Available KV cache memory: **12.14 GiB**"
- "GPU KV cache size: **99,440 tokens**"

That's 12.14 GiB / 99,440 = **128 KiB per token**. This checks out analytically:

- Llama-3.1-8B: 32 layers, 8 KV heads (GQA), 128 head dim
- Per token: 2 (K+V) × 8 heads × 128 dim × 2 bytes (FP16) × 32 layers = 131,072 bytes = 128 KiB

## Full budget at each gpu_memory_utilization

L4 total: 23,034 MiB (from `nvidia-smi`).

| | g85 | g90 | g95 |
|---|---|---|---|
| vLLM budget | 19,579 MiB | 20,731 MiB | 21,882 MiB |
| Model weights | 5,499 MiB | 5,499 MiB | 5,499 MiB |
| CUDA graphs | 584 MiB | 584 MiB | 584 MiB |
| KV cache (remainder) | ~13,500 MiB | ~14,650 MiB | ~15,800 MiB |
| KV tokens | ~99k | ~108k | ~116k |
| **Headroom (outside vLLM)** | **3,455 MiB** | **2,303 MiB** | **1,152 MiB** |

The headroom is what's left for the CUDA context, PyTorch allocator, and **activation memory during inference**. Activation memory scales with batch size — at c=16 with long sequences, peak activation memory for a forward pass through the attention + FFN layers is roughly 1-2 GiB.

At g95, 1,152 MiB of headroom is not enough for activation memory at high concurrency. At g90, 2,303 MiB is borderline — survives low-concurrency/short-sequence workloads but OOMs on heavier ones. This is consistent with g90 being partially successful (35/80) and g95 completely failing (0/80). The crashes are during inference (not startup) because activation memory is allocated dynamically per forward pass.
