import contextlib
import torch


@contextlib.contextmanager
def capture(output_dir="./traces"):
    """Capture a torch.profiler trace to Chrome trace JSON.

    Yields the underlying torch.profiler.profile object.
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            output_dir, use_gzip=False
        ),
    ) as prof:
        yield prof
