"""Map trace module events onto architectural blocks and produce a summary."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from xprofiler.trace import ModuleEvent, Trace


def load_model_config(name: str) -> dict:
    """Load a model config from xprofiler/models/<name>.json."""
    config_path = Path(__file__).parent / "models" / f"{name}.json"
    if not config_path.exists():
        available = [p.stem for p in config_path.parent.glob("*.json")]
        raise FileNotFoundError(
            f"No model config '{name}'. Available: {', '.join(available)}"
        )
    with open(config_path) as f:
        return json.load(f)


@dataclass
class BlockStats:
    cpu_time_us: float = 0.0
    count: int = 0
    per_instance_us: dict[int, float] = field(default_factory=dict)


def _find_single_step(tree: list[ModuleEvent], root_module: str) -> list[ModuleEvent]:
    """Extract one forward pass from a generate() trace.

    generate() calls forward() multiple times (once per token).
    The root module event repeats — take the first one only.
    """
    for root in tree:
        if root.class_name == root_module:
            return [root]
    # Fallback: return all roots
    return tree


def summarize(trace: Trace, config: dict) -> dict:
    """Walk the module tree and assign time to architectural blocks.

    config is a model config dict with keys:
        root_module, layer_module, block_mapping, containers
    """
    root_module = config["root_module"]
    layer_module = config["layer_module"]
    block_mapping = config["block_mapping"]
    containers = set(config["containers"])

    full_tree = trace.module_tree()

    # Count total forward passes
    all_forward = [r for r in full_tree if r.class_name == root_module]
    num_steps = len(all_forward)

    # Scope to single step
    tree = _find_single_step(full_tree, root_module)
    stats: dict[str, BlockStats] = {}

    def classify(module: ModuleEvent, parent_block: str | None):
        """Assign this module to a block and recurse into children."""
        block = block_mapping.get(module.class_name, parent_block)

        # Only record time for non-container modules that have a mapping
        if module.class_name in block_mapping and module.class_name not in containers:
            if block not in stats:
                stats[block] = BlockStats()
            s = stats[block]
            s.cpu_time_us += module.dur
            s.count += 1
            s.per_instance_us[module.instance_id] = (
                s.per_instance_us.get(module.instance_id, 0) + module.dur
            )

        for child in module.children:
            classify(child, block)

    for root in tree:
        classify(root, None)

    # Get total time and model name
    if tree and tree[0].class_name == root_module:
        total_time = tree[0].dur
        model_name = tree[0].class_name
    else:
        total_time = sum(r.dur for r in tree)
        model_name = "unknown"

    # Unaccounted time
    accounted = sum(s.cpu_time_us for s in stats.values())
    if total_time > accounted:
        stats["other"] = BlockStats(cpu_time_us=total_time - accounted, count=1)

    # Layer count from unique layer_module instance IDs
    layer_ids = set()

    def collect_layer_ids(modules):
        for m in modules:
            if m.class_name == layer_module:
                layer_ids.add(m.instance_id)
            collect_layer_ids(m.children)

    collect_layer_ids(tree)
    num_layers = len(layer_ids)

    # Build output
    blocks_out = {}
    for name, s in sorted(stats.items()):
        entry = {
            "cpu_time_us": round(s.cpu_time_us, 1),
            "pct": round(100 * s.cpu_time_us / total_time, 1) if total_time else 0,
            "count": s.count,
        }
        if len(s.per_instance_us) > 1:
            entry["per_instance_us"] = {
                k: round(v, 1)
                for k, v in sorted(s.per_instance_us.items())
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
