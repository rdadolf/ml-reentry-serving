import argparse
import gzip
import json
import sys
from pathlib import Path


def load_trace(path):
    """Load a Chrome trace JSON file (plain or gzipped)."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    else:
        with open(path) as f:
            return json.load(f)


def cmd_summary(args):
    from xprofiler import summary, trace

    # Resolve block mapping
    mappings = {"llama": summary.LLAMA_BLOCKS}
    block_mapping = mappings.get(args.mapping)
    if block_mapping is None:
        print(f"Unknown mapping: {args.mapping}")
        print(f"Available: {', '.join(mappings.keys())}")
        sys.exit(1)

    t = trace.load(args.trace)
    result = summary.summarize(t, block_mapping)
    print(summary.to_json(result))


def cmd_load(args):
    trace = load_trace(args.trace)
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])

    # Filter to complete events (ph="X") that have a duration
    complete = [e for e in events if e.get("ph") == "X" and "dur" in e]

    categories = set()
    for e in complete:
        cat = e.get("cat", "")
        if cat:
            categories.add(cat)

    ts_values = [e["ts"] for e in complete]
    time_range_us = max(ts_values) - min(ts_values) if ts_values else 0

    print(f"Trace: {args.trace}")
    print(f"Total events: {len(events)}")
    print(f"Complete (ph=X) events: {len(complete)}")
    print(f"Time range: {time_range_us / 1000:.1f} ms")
    print(f"Categories: {', '.join(sorted(categories))}")

    # Top 10 by duration
    by_dur = sorted(complete, key=lambda e: e.get("dur", 0), reverse=True)
    print(f"\nTop 10 events by duration:")
    for e in by_dur[:10]:
        dur_ms = e["dur"] / 1000
        print(f"  {dur_ms:8.2f} ms  {e.get('cat', ''):>20s}  {e['name']}")


def main():
    parser = argparse.ArgumentParser(prog="xprofiler")
    sub = parser.add_subparsers(dest="command")

    p_load = sub.add_parser("load", help="Load and summarize a trace file")
    p_load.add_argument("trace", help="Path to Chrome trace JSON (.json or .json.gz)")

    p_summary = sub.add_parser("summary", help="Produce per-block architectural summary")
    p_summary.add_argument("trace", help="Path to Chrome trace JSON (.json or .json.gz)")
    p_summary.add_argument(
        "--mapping",
        default="llama",
        help="Block mapping to use (default: llama)",
    )

    args = parser.parse_args()
    if args.command == "load":
        cmd_load(args)
    elif args.command == "summary":
        cmd_summary(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
