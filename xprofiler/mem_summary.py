"""Quick memory stats from a profiler trace. Usage: python -m xprofiler.mem_summary <trace.json>"""

import json
import statistics
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    with open(path) as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    mem_gpu = [
        e for e in events
        if e.get("name") == "[memory]" and e["args"]["Device Type"] == 1
    ]
    mem_gpu.sort(key=lambda e: e["ts"])

    if not mem_gpu:
        print("No GPU memory events found. Was profile_memory enabled?")
        sys.exit(1)

    alloc = [e["args"]["Total Allocated"] / 1024 / 1024 for e in mem_gpu]
    reserved = [e["args"]["Total Reserved"] / 1024 / 1024 for e in mem_gpu]
    sizes = [e["args"]["Bytes"] for e in mem_gpu]

    print(f"Trace: {path.name}")
    print(f"GPU memory events: {len(mem_gpu)}")
    print()

    print("--- Total Allocated (MB) ---")
    print(f"  min: {min(alloc):.1f}  max: {max(alloc):.1f}  "
          f"avg: {statistics.mean(alloc):.1f}  std: {statistics.stdev(alloc):.1f}  "
          f"median: {statistics.median(alloc):.1f}")

    print("--- Total Reserved (MB) ---")
    print(f"  min: {min(reserved):.1f}  max: {max(reserved):.1f}  "
          f"avg: {statistics.mean(reserved):.1f}  std: {statistics.stdev(reserved):.1f}  "
          f"median: {statistics.median(reserved):.1f}")

    print()
    print("--- Allocation sizes (bytes) ---")
    allocs = [s for s in sizes if s > 0]
    frees = [s for s in sizes if s < 0]
    print(f"  allocations: {len(allocs)}  frees: {len(frees)}")
    print(f"  alloc min: {min(allocs)}  max: {max(allocs)}  "
          f"median: {statistics.median(allocs):.0f}  "
          f"avg: {statistics.mean(allocs):.0f}")

    # Distribution buckets
    buckets = [
        ("< 1 KB", 0, 1024),
        ("1-16 KB", 1024, 16384),
        ("16-256 KB", 16384, 262144),
        ("256 KB - 1 MB", 262144, 1048576),
        ("1-16 MB", 1048576, 16777216),
        ("> 16 MB", 16777216, float("inf")),
    ]
    print()
    print("  Size distribution:")
    for label, lo, hi in buckets:
        count = sum(1 for s in allocs if lo <= s < hi)
        pct = 100 * count / len(allocs) if allocs else 0
        print(f"    {label:>15s}: {count:6d} ({pct:5.1f}%)")

    # Top 10 largest allocations
    top = sorted(allocs, reverse=True)[:10]
    print()
    print("  Top 10 largest allocations:")
    for s in top:
        if s >= 1048576:
            print(f"    {s / 1048576:.1f} MB")
        elif s >= 1024:
            print(f"    {s / 1024:.1f} KB")
        else:
            print(f"    {s} B")


if __name__ == "__main__":
    main()
