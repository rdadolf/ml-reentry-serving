"""Stretch goal: capture a trace with torch.compile + CUDA graphs (reduce-overhead)."""

import glob
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from xprofiler import capture

MODEL_ID = "TinyLlama/TinyLlama-1.1B-step-50K-105b"
OUTPUT_DIR = "./traces"
PROMPT = "The quick brown fox"


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).cuda()

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    print("Compiling model with mode='reduce-overhead' (CUDA graphs)...")
    model = torch.compile(model, mode="reduce-overhead")

    # Warmup: run a few forward passes to trigger compilation + graph recording
    print("Warming up (this may take a minute)...")
    with torch.no_grad():
        for i in range(3):
            _ = model(**inputs)
            print(f"  warmup {i+1}/3 done")
    torch.cuda.synchronize()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Running compiled forward pass under profiler...")
    with capture(output_dir=OUTPUT_DIR):
        with torch.no_grad():
            _ = model(**inputs)

    # Find the trace file that was just written
    traces = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
    if not traces:
        print("ERROR: No trace file produced!")
        return

    trace_path = traces[-1]
    print(f"\nTrace written: {trace_path}")
    print(f"Size: {os.path.getsize(trace_path) / 1024:.0f} KB")

    # Quick sanity check
    print()
    from xprofiler.__main__ import load_trace

    trace = load_trace(trace_path)
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    complete = [e for e in events if e.get("ph") == "X" and "dur" in e]

    cuda_events = [e for e in complete if e.get("cat") == "kernel"]
    cpu_events = [e for e in complete if e.get("cat") != "kernel"]

    print(f"Total events: {len(events)}")
    print(f"CPU events: {len(cpu_events)}")
    print(f"CUDA kernel events: {len(cuda_events)}")

    if cuda_events:
        print("\nCUDA profiling confirmed working.")
        top_kernels = sorted(cuda_events, key=lambda e: e.get("dur", 0), reverse=True)
        print("Top 5 CUDA kernels by duration:")
        for e in top_kernels[:5]:
            print(f"  {e['dur'] / 1000:8.2f} ms  {e['name']}")
    else:
        print("\nWARNING: No CUDA kernel events found. GPU profiling may not be working.")


if __name__ == "__main__":
    main()
