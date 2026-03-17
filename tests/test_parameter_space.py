import sys
sys.path.insert(0, "exp/vllm-sweeps")

import yaml
from pathlib import Path

# Import ParameterSpace by executing run-sweep.py up to the class definitions.
# This avoids importing mlflow (which needs a server) at test time.
_source = Path("exp/vllm-sweeps/run-sweep.py").read_text()
_code = _source.split("# ---------------------------------------------------------------------------\n# vLLM server lifecycle")[0]
exec(compile(_code, "run-sweep.py", "exec"))


def load_test_config():
    return yaml.safe_load(Path("exp/vllm-sweeps/sweep-config.yaml").read_text())


def test_total_matches_iteration_count():
    sweep = ParameterSpace(load_test_config())
    items = list(sweep)
    assert len(items) == len(sweep)


def test_all_combos_valid():
    """Every yielded combo must have input_len + output_len < max_model_len."""
    sweep = ParameterSpace(load_test_config())
    for params in sweep:
        assert params["input_len"] + params["output_len"] < params["max_model_len"], (
            f"Invalid: in={params['input_len']} out={params['output_len']} "
            f"ml={params['max_model_len']}"
        )


def test_no_duplicate_pairs_across_bands():
    """Each (input_len, output_len) pair appears in exactly one band."""
    sweep = ParameterSpace(load_test_config())
    all_pairs = []
    for band in sweep.band_pairs:
        all_pairs.extend(band)
    assert len(all_pairs) == len(set(all_pairs))


def test_band_assignment_is_minimal():
    """Each pair is assigned to the smallest max_model_len that fits."""
    sweep = ParameterSpace(load_test_config())
    for band_index, max_len in enumerate(sweep.max_model_lens):
        for input_len, output_len in sweep.band_pairs[band_index]:
            total = input_len + output_len
            for smaller in sweep.max_model_lens[:band_index]:
                assert total >= smaller, (
                    f"({input_len}, {output_len}) fits in {smaller} "
                    f"but was assigned to {max_len}"
                )


def test_run_names_unique():
    sweep = ParameterSpace(load_test_config())
    names = [ParameterSpace.run_name(p) for p in sweep]
    assert len(names) == len(set(names))


def test_server_config_grouping():
    """Server configs should be contiguous — no unnecessary restarts."""
    sweep = ParameterSpace(load_test_config())
    prev = None
    transitions = []
    for params in sweep:
        sc = sweep.server_config(params)
        if sc != prev:
            transitions.append(sc)
            prev = sc
    # Each unique server config should appear exactly once in transitions
    assert len(transitions) == len(set(map(lambda d: tuple(sorted(d.items())), transitions)))


def test_islice_resume():
    """Resuming via islice should produce correct remaining items."""
    import itertools
    sweep = ParameterSpace(load_test_config())
    items = list(sweep)
    skip = 100
    resumed = list(itertools.islice(sweep, skip, None))
    assert len(resumed) == len(items) - skip
    assert resumed[0] == items[skip]
    assert resumed[-1] == items[-1]
