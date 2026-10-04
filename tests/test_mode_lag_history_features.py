import numpy as np

from eval.experiment_death_aware_mode_lag_features import history


def _data(times, modes):
    return {
        "base_mode": np.asarray(modes, dtype=np.int64),
        "game_ids": np.full(len(times), 42, dtype=np.int64),
        "sides": np.asarray(["R"] * len(times)),
        "issued_at": np.asarray(times, dtype=np.float32),
    }


def test_run_length_counts_adjacent_same_mode_samples():
    # Input order is intentionally shuffled; history must sort by timestamp.
    d = _data([9, 0, 6, 3, 12], [1, 1, 1, 1, 2])
    z = history(d)
    at = {int(t): i for i, t in enumerate(d["issued_at"])}
    assert z[at[0], 4] == 1
    assert z[at[3], 4] == 2
    assert z[at[6], 4] == 3
    assert z[at[9], 4] == 4
    assert z[at[12], 4] == 1


def test_missing_timestamp_breaks_run_and_exact_lag():
    d = _data([0, 3, 9, 12], [1, 1, 1, 1])  # t=6 is absent
    z = history(d)
    at = {int(t): i for i, t in enumerate(d["issued_at"])}
    assert z[at[9], 4] == 1  # a gap cannot be counted as an adjacent sample
    assert z[at[12], 4] == 2
    assert z[at[9], 0] == -1  # exact t=6 lag is unknown
    assert z[at[9], 1] == 1  # exact t=3 lag remains usable


def test_feature_at_time_is_not_affected_by_future_rows():
    prefix = _data([0, 3, 6], [0, 0, 1])
    extended = _data([0, 3, 6, 9, 12], [0, 0, 1, 2, 2])
    a = history(prefix)
    b = history(extended)
    np.testing.assert_array_equal(a, b[:3])
