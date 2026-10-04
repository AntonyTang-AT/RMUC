import json
from pathlib import Path

import pytest


def test_production_report_position_policy_matches_manifest():
    manifest_path = Path("out_m2/entity_motion_lightgbm/production_manifest.json")
    report_path = Path("out_m2/production_ensemble_eval/report.json")
    if not manifest_path.exists() or not report_path.exists():
        pytest.skip("production ensemble artifacts have not been generated yet")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    policy = report["runtime_policy"]["entity_position"]

    for horizon in ("10", "30", "60", "120"):
        enabled = bool(manifest["horizons"][horizon]["position_enabled"])
        expected_source = "entity_motion_lightgbm" if enabled else "neural"
        assert policy[horizon] == expected_source
        assert report["entity"][horizon]["position"]["position_sidecar_enabled"] == enabled
