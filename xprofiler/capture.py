import contextlib
import os
from datetime import datetime

import torch


@contextlib.contextmanager
def capture(output_dir="./traces", trace_name=None, profile_memory=True,
            record_shapes=False):
    """Capture a torch.profiler trace to Chrome trace JSON.

    Args:
        output_dir: Directory to write trace files.
        trace_name: Base name for the trace file (without extension).
            If None, uses a timestamp-based default.
        profile_memory: Track CUDA memory allocations/frees.
        record_shapes: Record tensor shapes for each op.

    Yields the underlying torch.profiler.profile object.
    Sets prof.trace_path after tracing completes.
    """
    if trace_name is None:
        trace_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, f"{trace_name}.trace.json")

    def on_trace_ready(prof):
        prof.export_chrome_trace(trace_path)
        prof.trace_path = trace_path

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        profile_memory=profile_memory,
        record_shapes=record_shapes,
        with_flops=record_shapes,
        on_trace_ready=on_trace_ready,
    ) as prof:
        yield prof
