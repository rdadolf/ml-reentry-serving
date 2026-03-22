"""Load Chrome trace JSON and build a containment tree from nn.Module events."""

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModuleEvent:
    """An nn.Module trace event with its children."""

    name: str  # e.g. "nn.Module: LlamaAttention_0"
    class_name: str  # e.g. "LlamaAttention"
    instance_id: int  # e.g. 0
    ts: float  # start timestamp (us)
    dur: float  # duration (us)
    children: list["ModuleEvent"] = field(default_factory=list)


@dataclass
class Trace:
    """A loaded trace with indexed events."""

    raw_events: list[dict]

    @property
    def module_events(self) -> list[dict]:
        return [
            e
            for e in self.raw_events
            if e.get("name", "").startswith("nn.Module:") and e.get("ph") == "X"
        ]

    @property
    def kernel_events(self) -> list[dict]:
        return [
            e
            for e in self.raw_events
            if e.get("cat") == "kernel" and e.get("ph") == "X"
        ]

    @property
    def cpu_op_events(self) -> list[dict]:
        return [
            e
            for e in self.raw_events
            if e.get("cat") == "cpu_op" and e.get("ph") == "X"
        ]

    def module_tree(self) -> list[ModuleEvent]:
        """Build a containment tree from nn.Module events using timestamp nesting."""
        modules = []
        for e in self.module_events:
            name = e["name"]
            # "nn.Module: LlamaAttention_0" -> class_name="LlamaAttention", id=0
            label = name.split(": ", 1)[1]
            parts = label.rsplit("_", 1)
            class_name = parts[0]
            instance_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            modules.append(
                ModuleEvent(
                    name=name,
                    class_name=class_name,
                    instance_id=instance_id,
                    ts=e["ts"],
                    dur=e["dur"],
                )
            )

        # Sort by start time, then longest duration first (parents before children)
        modules.sort(key=lambda m: (m.ts, -m.dur))

        # Build tree via stack-based containment
        roots = []
        stack = []  # stack of (module, end_time)

        for m in modules:
            end = m.ts + m.dur
            # Pop anything from the stack that doesn't contain this event
            while stack and stack[-1][1] < m.ts:
                stack.pop()

            if stack:
                stack[-1][0].children.append(m)
            else:
                roots.append(m)

            stack.append((m, end))

        return roots

    def ops_within(self, module: ModuleEvent) -> list[dict]:
        """Return all cpu_op events whose timestamps fall within a module's span."""
        end = module.ts + module.dur
        return [
            e
            for e in self.cpu_op_events
            if e["ts"] >= module.ts and e["ts"] + e.get("dur", 0) <= end
        ]


def load(path: str | Path) -> Trace:
    """Load a Chrome trace JSON file (plain or gzipped)."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            data = json.load(f)
    else:
        with open(path) as f:
            data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    return Trace(raw_events=events)
