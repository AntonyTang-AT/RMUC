from pathlib import Path

import numpy as np
import pytest


def test_entity_motion_sidecar_loads_and_returns_all_horizons():
    manifest = Path("out_m2/entity_motion_lightgbm/production_manifest.json")
    if not manifest.exists():
        pytest.skip("locked-test entity sidecar has not been trained yet")
    from forecast.entity_motion_lgbm import EntityMotionLGBMEnsemble

    ensemble = EntityMotionLGBMEnsemble(manifest)
    row = np.zeros(232, dtype=np.float32)
    # Mark one tracked slot with a valid normalized position and presence.
    row[0:2] = (0.25, 0.5)
    row[3] = 1.0
    row[9] = 1.0
    result = ensemble.predict([row, row.copy()])
    assert set(result) == {"10", "30", "60", "120"}
    for item in result.values():
        assert item["positions"].shape == (16, 2)
        assert len(item["functional_zones"]) == 16
        assert len(item["canonical_zones"]) == 16
