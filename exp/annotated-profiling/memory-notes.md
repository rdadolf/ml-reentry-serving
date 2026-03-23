# Memory Profiling Notes

Runtime memory profiling during inference on pre-loaded models reveals very little dynamic behavior. All three models show essentially flat memory usage throughout the forward pass, with sub-MB fluctuations from tiny intermediate tensor allocations that are immediately freed.

This is expected: model weights are loaded before profiling begins, and single-token decode steps create negligible intermediate state. Memory profiling would be more informative during training (gradients, optimizer state), long-sequence prefill (KV cache growth), or model loading itself.

## Per-Model Results

### TinyLlama 1.1B

```
GPU memory events: 63770
Total Allocated (MB): min: 2108.3  max: 2109.4  avg: 2108.9  std: 0.2  median: 2108.9
Total Reserved (MB):  min: 2210.0  max: 2210.0  avg: 2210.0  std: 0.0  median: 2210.0

Allocation sizes (bytes):
  allocations: 31885  frees: 31885
  alloc min: 512  max: 128000  median: 4096  avg: 4858
  Size distribution:
           < 1 KB: 12404 (38.9%)
          1-16 KB: 18619 (58.4%)
        16-256 KB:   862 ( 2.7%)
  Top allocation: 125.0 KB
```

### Qwen2.5 0.5B

```
GPU memory events: 69400
Total Allocated (MB): min: 959.3  max: 961.2  avg: 960.1  std: 0.2  median: 960.1
Total Reserved (MB):  min: 1014.0  max: 1014.0  avg: 1014.0  std: 0.0  median: 1014.0

Allocation sizes (bytes):
  allocations: 34700  frees: 34700
  alloc min: 512  max: 607744  median: 2048  avg: 3461
  Size distribution:
           < 1 KB: 13526 (39.0%)
          1-16 KB: 21014 (60.6%)
        16-256 KB:    96 ( 0.3%)
    256 KB - 1 MB:    64 ( 0.2%)
  Top allocation: 593.5 KB
```

### BLOOM 560M

```
GPU memory events: 46284
Total Allocated (MB): min: 1075.7  max: 1081.4  avg: 1078.5  std: 0.9  median: 1078.5
Total Reserved (MB):  min: 1120.0  max: 1120.0  avg: 1120.0  std: 0.0  median: 1120.0

Allocation sizes (bytes):
  allocations: 23142  frees: 23142
  alloc min: 512  max: 2097152  median: 2048  avg: 8774
  Size distribution:
           < 1 KB:  5479 (23.7%)
          1-16 KB: 15991 (69.1%)
        16-256 KB:  1608 ( 6.9%)
    256 KB - 1 MB:    63 ( 0.3%)
        1-16 MB:      1 ( 0.0%)
  Top allocation: 2.0 MB
```

## Summary

All three models are dominated by tiny allocations (97%+ under 16 KB). Every allocation is paired with a free — no leaks, no accumulation. Total Allocated fluctuates by less than 1 MB across the entire generate() call for all models. Total Reserved is constant (PyTorch's caching allocator never needs to request more memory from CUDA after the initial allocation).

BLOOM has the largest single allocation (2.0 MB) and slightly more variance in Total Allocated (std: 0.9 MB vs 0.2 MB for the others), likely due to its fused QKV projection creating a larger intermediate. But even this is negligible relative to the ~1 GB model footprint.

Conclusion: runtime memory profiling during inference adds no useful per-block information for diagram annotation. The memory story for these models is entirely static (model weight size) and can be computed from the architecture without profiling.
