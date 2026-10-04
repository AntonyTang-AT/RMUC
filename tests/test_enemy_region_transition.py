import json

import numpy as np
import pytest

from forecast.enemy_region_transition import EnemyRegionTransitionEnsemble
from eval.train_enemy_region_transition import export_frozen


def test_transition_sidecar_uses_frozen_key_and_persistence_blend(tmp_path):
    manifest = {
        "schema": "enemy-region-transition-manifest-v1",
        "alpha_transition": 0.8,
        "laplace_smoothing": 10.0,
        "time_bucket_seconds": 60,
        "enabled": True,
        "table": {
            json.dumps([1, "红", 2, 0], ensure_ascii=False, separators=(",", ":")):
                [0.0, 0.0, 1.0] + [0.0] * 9,
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    result = EnemyRegionTransitionEnsemble(path).predict(
        base_region=1, side="红", issued_at=125.0, base_mode=0)

    assert result["top"] == "x0y2"
    assert result["model_vs_baseline"]["baseline_top"] == "x0y1"
    assert result["model_vs_baseline"]["transition_key_matched"] is True
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_transition_sidecar_falls_back_to_uniform_when_key_is_absent(tmp_path):
    manifest = {
        "schema": "enemy-region-transition-manifest-v1",
        "alpha_transition": 0.8,
        "laplace_smoothing": 10.0,
        "time_bucket_seconds": 60,
        "enabled": True,
        "table": {},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = EnemyRegionTransitionEnsemble(path).predict(
        base_region=4, side="蓝", issued_at=0.0, base_mode=1)

    assert result["top"] == "x1y1"
    assert result["model_vs_baseline"]["transition_key_matched"] is False


def test_transition_sidecar_reports_manifest_horizon(tmp_path):
    manifest = {
        "schema": "enemy-region-transition-manifest-v1",
        "horizon_seconds": 10,
        "alpha_transition": 0.9,
        "laplace_smoothing": 10.0,
        "time_bucket_seconds": 60,
        "enabled": True,
        "table": {},
    }
    path = tmp_path / "manifest_10s.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    result = EnemyRegionTransitionEnsemble(path).predict(
        base_region=1, side="红", issued_at=125.0, base_mode=0)

    assert result["source"] == "enemy_region_transition_10s_locked_test_pass"


def test_frozen_export_reads_train_only_and_requires_double_metric_gate(tmp_path):
    root = tmp_path / "splits"
    root.mkdir()
    np.savez(
        root / "train.npz",
        region_y=np.asarray([[2, 3, 4, 5]], dtype=np.int64),
        base_region=np.asarray([1], dtype=np.int64),
        sides=np.asarray(["红"]),
        issued_at=np.asarray([125.0]),
        base_mode=np.asarray([0], dtype=np.int64),
    )
    evidence = {
        "candidate": {
            "validation": {"accuracy": 0.8, "macro_f1": 0.5},
            "locked_test": {"accuracy": 0.75, "macro_f1": 0.4},
        },
        "persistence": {
            "validation": {"accuracy": 0.79, "macro_f1": 0.49},
            "locked_test": {"accuracy": 0.74, "macro_f1": 0.39},
        },
    }
    output = tmp_path / "manifest.json"

    export_frozen(root, output, 10, 0.9, evidence)

    manifest = json.loads(output.read_text(encoding="utf-8"))
    key = json.dumps([1, "红", 2, 0], ensure_ascii=False, separators=(",", ":"))
    assert manifest["enabled"] is True
    assert manifest["horizon_seconds"] == 10
    assert manifest["table"][key][2] > manifest["table"][key][3]

    evidence["candidate"]["locked_test"]["macro_f1"] = evidence[
        "persistence"]["locked_test"]["macro_f1"]
    with pytest.raises(ValueError, match="locked_test double-metric"):
        export_frozen(root, tmp_path / "rejected.json", 10, 0.9, evidence)
