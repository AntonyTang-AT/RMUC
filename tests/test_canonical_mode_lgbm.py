import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from forecast.canonical_mode_lgbm import CanonicalModeLGBMEnsemble


def test_locked_test_manifest_contains_all_horizons_and_decodes_probabilities():
    path = os.path.join(os.path.dirname(__file__), '..', 'out_m2',
                        'canonical_mode_lightgbm', 'production_manifest.json')
    if not os.path.exists(path):
        return
    expert = CanonicalModeLGBMEnsemble(path)
    result = expert.predict([np.zeros(232, dtype=np.float32)], 0.0, 1)
    assert set(result) == {'10', '30', '60', '120'}
    for row in result.values():
        assert set(row['probabilities']) == {'进攻', '均衡', '防守'}
        assert abs(sum(row['probabilities'].values()) - 1.0) < 2e-5
        assert row['source'] == 'canonical_mode_lightgbm_locked_test_pass'

