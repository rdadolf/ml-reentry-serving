"""Map trace module events onto architectural blocks and produce a summary."""

import json
from dataclasses import dataclass, field

from xprofiler.trace import ModuleEvent, Trace

# TOOD: This is still a bit hardcoded for LLaMA-based models. Some warts:
# - L93: endswith("ForCausalLM") — assumes the root model class name
# - L112: "LlamaDecoderLayer" literal — hardcoded for layer counting
# - L115: fallback "DecoderLayer" substring check — slightly more general but still assumes naming
# - The "decoder_layer" magic string in the block mapping that gets special-cased to "skip this, it's a container"
# - No norm splitting (pre-attn vs pre-FFN)
# - No residual time estimation


# Block mappings: dict of module class name -> block label.
# The summarizer walks the module tree top-down and assigns each module
# to the first matching block. Unmatched modules fall through to their
# parent's block, or "other" if at root level.

LLAMA_BLOCKS = {
    "LlamaAttention": "attention",
    "LlamaMLP": "mlp",
    "LlamaRMSNorm": "rms_norm",
    "LlamaRotaryEmbedding": "rope",
    "LlamaDecoderLayer": "decoder_layer",
    "Embedding": "embedding",
}


@dataclass
class BlockStats:
    gpu_time_us: float = 0.0
    cpu_time_us: float = 0.0
    count: int = 0
    per_instance_gpu_us: dict[int, float] = field(default_factory=dict)


def _find_single_step(tree: list[ModuleEvent]) -> list[ModuleEvent]:
    """Extract one forward pass from a generate() trace.

    generate() calls forward() multiple times (once per token).
    The ForCausalLM module event repeats — take the first one only,
    which includes the prefill + first decode step.
    """
    for root in tree:
        if root.class_name.endswith("ForCausalLM"):
            # The ForCausalLM node appears once per forward() call.
            # Return just this one as the root.
            return [root]
    # No ForCausalLM found — return all roots as-is
    return tree


def summarize(trace: Trace, block_mapping: dict[str, str]) -> dict:
    """Walk the module tree and assign time to architectural blocks.

    Scopes to a single forward pass if the trace contains multiple
    (e.g., from generate()). Returns a dict suitable for JSON serialization.
    """
    full_tree = trace.module_tree()

    # Count total forward passes for context
    all_forward = [
        r for r in full_tree if r.class_name.endswith("ForCausalLM")
    ]
    num_steps = len(all_forward)

    # Scope to single step
    tree = _find_single_step(full_tree)
    stats: dict[str, BlockStats] = {}

    def classify(module: ModuleEvent, parent_block: str | None):
        """Assign this module to a block and recurse into children."""
        block = block_mapping.get(module.class_name, parent_block)

        # If this module maps to a block (directly, not inherited),
        # record its time. Skip "decoder_layer" — it's a container,
        # its children carry the actual block assignments.
        if module.class_name in block_mapping and block != "decoder_layer":
            if block not in stats:
                stats[block] = BlockStats()
            s = stats[block]
            s.cpu_time_us += module.dur
            s.count += 1
            s.per_instance_gpu_us[module.instance_id] = (
                s.per_instance_gpu_us.get(module.instance_id, 0) + module.dur
            )

        # Recurse — children inherit this module's block if they don't
        # have their own mapping
        for child in module.children:
            classify(child, block)

    # Walk scoped tree
    for root in tree:
        classify(root, None)

    # Get total time and model name
    if tree and tree[0].class_name.endswith("ForCausalLM"):
        total_time = tree[0].dur
        model_name = tree[0].class_name
    else:
        total_time = sum(r.dur for r in tree)
        model_name = "unknown"

    # Assign unaccounted time to "other"
    accounted = sum(s.cpu_time_us for s in stats.values())
    if total_time > accounted:
        stats["other"] = BlockStats(
            cpu_time_us=total_time - accounted, count=1
        )

    # Detect layer count from unique DecoderLayer instance IDs
    decoder_ids = set()

    def collect_decoder_ids(modules):
        for m in modules:
            if "DecoderLayer" in m.class_name:
                decoder_ids.add(m.instance_id)
            collect_decoder_ids(m.children)

    collect_decoder_ids(tree)
    num_layers = len(decoder_ids)

    # Build output
    blocks_out = {}
    for name, s in sorted(stats.items()):
        entry = {
            "cpu_time_us": round(s.cpu_time_us, 1),
            "pct": round(100 * s.cpu_time_us / total_time, 1) if total_time else 0,
            "count": s.count,
        }
        if len(s.per_instance_gpu_us) > 1:
            entry["per_instance_us"] = {
                k: round(v, 1)
                for k, v in sorted(s.per_instance_gpu_us.items())
            }
        blocks_out[name] = entry

    return {
        "model": model_name,
        "num_layers": num_layers,
        "num_steps_in_trace": num_steps,
        "total_time_us": round(total_time, 1),
        "blocks": blocks_out,
    }


def to_json(result: dict, indent: int = 2) -> str:
    return json.dumps(result, indent=indent)
