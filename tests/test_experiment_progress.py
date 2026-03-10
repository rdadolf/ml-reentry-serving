"""Tests for ExperimentProgress tracker."""

import json
import sys
from pathlib import Path

import pytest

# experiment_progress lives in exp/vllm-sweeps/, add it to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp" / "vllm-sweeps"))

from experiment_progress import ExperimentProgress


@pytest.fixture(autouse=True)
def _reset_progress():
    """Reset class state before each test."""
    ExperimentProgress.reset()
    yield
    ExperimentProgress.reset()


def read_progress(path: Path) -> dict:
    return json.loads(path.read_text())


# ── init + step ──────────────────────────────────────────────────────


class TestInitAndStep:
    def test_init_writes_file(self, tmp_path):
        p = tmp_path / "progress.json"
        ExperimentProgress.init(5, p)
        assert read_progress(p) == {"completed": 0, "total": 5}

    def test_step_increments(self, tmp_path):
        p = tmp_path / "progress.json"
        ExperimentProgress.init(3, p)
        ExperimentProgress.step()
        assert read_progress(p) == {"completed": 1, "total": 3}
        ExperimentProgress.step()
        ExperimentProgress.step()
        assert read_progress(p) == {"completed": 3, "total": 3}

    def test_init_without_path_does_not_write(self, tmp_path):
        ExperimentProgress.init(5)
        # No file should exist anywhere — just confirm no error
        assert ExperimentProgress._completed == 0
        assert ExperimentProgress._total == 5
        assert ExperimentProgress._path is None

    def test_init_resets_completed(self, tmp_path):
        p = tmp_path / "progress.json"
        ExperimentProgress.init(3, p)
        ExperimentProgress.step()
        ExperimentProgress.step()
        ExperimentProgress.init(10, p)
        assert read_progress(p) == {"completed": 0, "total": 10}


# ── track ────────────────────────────────────────────────────────────


class TestTrack:
    def test_track_with_list(self, tmp_path):
        p = tmp_path / "progress.json"
        items = ["a", "b", "c"]
        result = list(ExperimentProgress.track(items, path=p))
        assert result == ["a", "b", "c"]
        assert read_progress(p) == {"completed": 3, "total": 3}

    def test_track_with_explicit_total(self, tmp_path):
        p = tmp_path / "progress.json"

        def gen():
            yield 1
            yield 2

        result = list(ExperimentProgress.track(gen(), path=p, total=2))
        assert result == [1, 2]
        assert read_progress(p) == {"completed": 2, "total": 2}

    def test_track_without_path(self):
        """track() without path should work (no file written)."""
        items = [1, 2, 3]
        result = list(ExperimentProgress.track(items))
        assert result == [1, 2, 3]
        assert ExperimentProgress._completed == 3

    def test_track_steps_after_each_yield(self, tmp_path):
        """step() fires after yield returns, so inside the loop body
        the completed count reflects the *previous* iteration's step."""
        p = tmp_path / "progress.json"
        items = ["a", "b", "c"]
        snapshots_before = []
        for item in ExperimentProgress.track(items, path=p):
            snapshots_before.append(read_progress(p)["completed"])
        # Inside the loop, we see 0, 1, 2 (step hasn't fired yet for current item)
        assert snapshots_before == [0, 1, 2]
        # After the loop, all steps have fired
        assert read_progress(p) == {"completed": 3, "total": 3}

    def test_track_generator_without_len_requires_total(self):
        def gen():
            yield 1

        with pytest.raises(TypeError):
            # Generator has no len(), so this should fail
            list(ExperimentProgress.track(gen()))


# ── takes_step decorator ─────────────────────────────────────────────


class TestTakesStep:
    def test_decorator_increments(self, tmp_path):
        p = tmp_path / "progress.json"
        ExperimentProgress.init(3, p)

        @ExperimentProgress.takes_step
        def do_work(x):
            return x * 2

        assert do_work(1) == 2
        assert read_progress(p) == {"completed": 1, "total": 3}
        do_work(2)
        do_work(3)
        assert read_progress(p) == {"completed": 3, "total": 3}

    def test_decorator_preserves_function_name(self):
        @ExperimentProgress.takes_step
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_decorator_without_path(self):
        """Decorator works even without a file path (no-op write)."""
        ExperimentProgress.init(2)

        @ExperimentProgress.takes_step
        def work():
            pass

        work()
        work()
        assert ExperimentProgress._completed == 2


# ── fsync / crash safety ─────────────────────────────────────────────


class TestFileSafety:
    def test_file_is_valid_json_after_each_step(self, tmp_path):
        """Each write produces valid, complete JSON."""
        p = tmp_path / "progress.json"
        ExperimentProgress.init(100, p)
        for _ in range(10):
            ExperimentProgress.step()
            # Should always be parseable
            data = json.loads(p.read_text())
            assert "completed" in data
            assert "total" in data

    def test_file_is_overwritten_not_appended(self, tmp_path):
        p = tmp_path / "progress.json"
        ExperimentProgress.init(5, p)
        ExperimentProgress.step()
        ExperimentProgress.step()
        content = p.read_text()
        # Should be a single JSON object, not multiple
        assert content.count("{") == 1
