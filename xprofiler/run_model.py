"""Capture a trace for any configured model and run a summary."""

import argparse
import os
import sys
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from xprofiler import capture
from xprofiler.summary import load_model_config

PROMPT = "The quick brown fox"
MAX_NEW_TOKENS = 32
OUTPUT_DIR = "./traces"


def _param_count_str(model):
    """Format parameter count as e.g. '1.1B', '560M'."""
    n = sum(p.numel() for p in model.parameters())
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    return f"{n / 1e6:.0f}M"


def main():
    parser = argparse.ArgumentParser(prog="xprofiler.run_model")
    parser.add_argument("model", nargs="?", default="llama", help="Model config name")
    parser.add_argument("--mem", action=argparse.BooleanOptionalAction, default=True,
                        help="Profile memory (default: on)")
    parser.add_argument("--shapes", action=argparse.BooleanOptionalAction, default=False,
                        help="Record tensor shapes (default: off)")
    args = parser.parse_args()

    model_name = args.model

    try:
        config = load_model_config(model_name)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    model_id = config["model_id"]
    dtype = getattr(torch, config["dtype"])

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).cuda()

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize()

    # Build trace name: <model-name>-<param-count>-<YYYYMMDD-HHMMSS>
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trace_name = f"{model_name}-{_param_count_str(model)}-{timestamp}"

    print("Running generate() under profiler...")
    with capture(output_dir=OUTPUT_DIR, trace_name=trace_name,
                 profile_memory=args.mem, record_shapes=args.shapes) as prof:
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    trace_path = prof.trace_path
    print(f"\nTrace written: {trace_path}")
    print(f"Size: {os.path.getsize(trace_path) / 1024:.0f} KB")

    # Run summary and save
    from xprofiler import summary, trace

    t = trace.load(trace_path)
    result = summary.summarize(t, config)

    summary_path = trace_path.replace(".trace.json", ".summary.json")
    with open(summary_path, "w") as f:
        f.write(summary.to_json(result))

    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
