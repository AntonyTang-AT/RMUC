"""Output calibration for causal forecasts across horizons.

The forecaster keeps the same label set at every horizon.  This module only
widens a later horizon's probability distribution when it is more confident
than an earlier horizon for the same task and sample.  It never removes a
class or changes the argmax unless the original probabilities are tied.
"""

from __future__ import annotations

import numpy as np


CALIBRATION_VERSION = "monotonic_horizon_v2_temperature"
HORIZON_TEMPERATURES = {10: 1.00, 30: 1.08, 60: 1.18, 120: 1.32}


def enforce_monotonic_horizon_confidence(probabilities_by_horizon):
    """Return distributions whose top probability never increases with time.

    ``probabilities_by_horizon`` maps each horizon to a one-dimensional
    probability vector for one task and one example.  Later vectors are mixed
    with the uniform distribution only as much as needed to bring their top
    probability below the previous horizon's top probability.
    """
    out = {}
    previous_max = None
    for horizon in sorted(probabilities_by_horizon):
        values = np.asarray(probabilities_by_horizon[horizon], dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("each horizon probability must be a non-empty vector")
        values = np.clip(values, 0.0, 1.0)
        total = float(values.sum())
        if total <= 0.0:
            values.fill(1.0 / values.size)
        else:
            values /= total
        current_max = float(values.max())
        if previous_max is not None and current_max > previous_max + 1e-12:
            uniform = 1.0 / values.size
            denominator = current_max - uniform
            if denominator > 1e-12:
                alpha = min(1.0, max(0.0, (current_max - previous_max) / denominator))
                values = (1.0 - alpha) * values + alpha * uniform
        previous_max = float(values.max())
        out[int(horizon)] = values.astype(np.float32).tolist()
    return out


def soften_horizon_probabilities(probabilities_by_horizon,
                                 temperatures=None):
    """Flatten farther-horizon distributions before monotonic calibration.

    This is a reporting/calibration operation, not a label change: a distant
    forecast keeps the same classes but expresses its larger set of plausible
    futures instead of presenting a spurious point certainty.
    """
    temperatures = temperatures or HORIZON_TEMPERATURES
    out = {}
    for horizon, values in probabilities_by_horizon.items():
        values = np.asarray(values, dtype=np.float64)
        values = np.clip(values, 1e-8, 1.0)
        values /= max(float(values.sum()), 1e-12)
        temperature = max(1.0, float(temperatures.get(int(horizon), 1.0)))
        logits = np.log(values) / temperature
        logits -= float(logits.max())
        values = np.exp(logits)
        values /= max(float(values.sum()), 1e-12)
        out[int(horizon)] = values.astype(np.float32).tolist()
    return out


def calibrate_horizon_probabilities(probabilities_by_horizon):
    """Apply horizon-aware uncertainty widening and monotonic confidence."""
    return enforce_monotonic_horizon_confidence(
        soften_horizon_probabilities(probabilities_by_horizon))
