import numpy as np
import pytest

from eval.evaluate_production_ensemble import _region_marginal_probs


def test_region_probabilities_collapse_to_normalized_axis_marginals():
    probabilities = np.zeros((2, 12), dtype=np.float64)
    probabilities[0, 0] = 0.25
    probabilities[0, 2] = 0.75
    probabilities[1, 7] = 0.4
    probabilities[1, 11] = 0.6

    result = _region_marginal_probs(probabilities)

    np.testing.assert_allclose(result['x_band'], [[1, 0, 0, 0], [0, 0, 0.4, 0.6]])
    np.testing.assert_allclose(result['y_band'], [[0.25, 0, 0.75], [0, 0.4, 0.6]])
    np.testing.assert_allclose(result['x_band'].sum(axis=1), 1.0)
    np.testing.assert_allclose(result['y_band'].sum(axis=1), 1.0)


def test_region_marginals_reject_non_twelve_cell_distributions():
    with pytest.raises(ValueError, match='expected region probabilities'):
        _region_marginal_probs(np.zeros((1, 11)))
