from pathlib import Path

import numpy as np
import pytest


def test_global_task_sidecar_loads_and_keeps_promoted_horizons_only():
    manifest = Path("out_m2/global_task_lightgbm/production_manifest.json")
    if not manifest.exists():
        pytest.skip("global task sidecar has not been trained yet")
    from forecast.global_task_lgbm import GlobalTaskLGBMEnsemble

    ensemble = GlobalTaskLGBMEnsemble(manifest)
    row = np.zeros(232, dtype=np.float32)
    result = ensemble.predict([row, row.copy()], 1.0, 1, -1)
    assert set(result) == {"10", "30", "60", "120"}
    assert "opponent_trend" in result["10"]
    assert "enemy_region" not in result["10"]
    assert "enemy_region" in result["120"]
