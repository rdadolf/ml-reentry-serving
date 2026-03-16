"""Generic experiment progress tracker that writes state to a JSON file.

Usage — iterable wrapper (tqdm-style):

    for item in ExperimentProgress.track(items, path="/results/progress.json"):
        do_work(item)

Usage — decorator:

    ExperimentProgress.init(total, "/results/progress.json")

    @ExperimentProgress.takes_step
    def do_work(item):
        ...

    for item in items:
        do_work(item)

Usage — manual:

    ExperimentProgress.init(total, path)
    for item in items:
        do_work(item)
        ExperimentProgress.step()

All state is class-level so decorators work without instance access.
"""

import json
import functools
import os
from pathlib import Path


class ExperimentProgress:
    _path: Path | None = None
    _total: int = 0
    _completed: int = 0

    @classmethod
    def reset(cls):
        """Reset all state. Mainly useful for tests."""
        cls._path = None
        cls._total = 0
        cls._completed = 0

    @classmethod
    def init(cls, total: int, path: str | Path | None = None):
        """Set the total number of steps and optional output file path."""
        cls._total = total
        cls._completed = 0
        if path is not None:
            cls._path = Path(path)
        cls._write()

    @classmethod
    def step(cls):
        """Mark one step as completed."""
        cls._completed += 1
        cls._write()

    @classmethod
    def track(cls, iterable, *, path: str | Path | None = None, total: int | None = None):
        """Wrap an iterable, stepping after each item yields.

        If total is not provided, tries len(iterable). If path is provided,
        initializes the tracker (equivalent to calling init() first).
        """
        if total is None:
            total = len(iterable)
        cls.init(total, path)
        for item in iterable:
            yield item
            cls.step()

    @classmethod
    def takes_step(cls, fn):
        """Decorator that calls step() after the wrapped function returns."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            cls.step()
            return result
        return wrapper

    @classmethod
    def _write(cls):
        if cls._path is None:
            return
        data = {"completed": cls._completed, "total": cls._total}
        fd = os.open(str(cls._path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            os.write(fd, json.dumps(data).encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    @classmethod
    def load_completed(cls, path: str | Path) -> int:
        """Read completed count from an existing progress file."""
        p = Path(path)
        if not p.exists():
            return 0
        try:
            data = json.loads(p.read_text())
            return data.get("completed", 0)
        except (json.JSONDecodeError, OSError):
            return 0

    @classmethod
    def set_completed(cls, n: int):
        """Set the completed count (e.g. when resuming)."""
        cls._completed = n
        cls._write()
