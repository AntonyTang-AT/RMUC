import numpy as np
import pytest

from forecast.mode_change_xgb import (
    BASE_FEATURE_DIM,
    FEATURE_DIM,
    build_latest_mode_change_features,
    build_mode_change_features,
    observable_snapshot_base_features,
)
import runtime.temporal_forecast as temporal_runtime
from runtime.temporal_forecast import apply_mode_xgb_candidate


def _snapshot(t, *, side="红", drop_fields=False):
    hero = {"entity_type": "英雄", "position": [5 + t / 100, 7],
            "hp": 100, "maxhp": 100, "confidence": .9}
    opp = {"entity_type": "步兵3", "position": [20 - t / 100, 7],
           "hp": 80, "confidence": .8}
    if drop_fields:
        hero.pop("position")
        hero.pop("hp")
        opp.pop("confidence")
    return {
        "side": side, "issued_at": float(t),
        "observable": {
            "self_entities": {"self-hero": hero, "self-base": {
                "entity_type": "基地", "hp": 5000, "maxhp": 5000}},
            "opponent_entities": {"opp-infantry": opp},
        },
        # Deliberately irrelevant: target labels must not enter online inputs.
        "targets": {"30": {"truth": {"mode": "防守"}}},
    }


def test_snapshot_builder_is_a_20d_visible_only_causal_contract():
    first = _snapshot(0)
    second = _snapshot(3, drop_fields=True)
    v0, m0 = observable_snapshot_base_features(first)
    v1, m1 = observable_snapshot_base_features(second, 0)
    assert v0.shape == v1.shape == (BASE_FEATURE_DIM,)
    assert v0.dtype == np.float32 and v1.dtype == np.float32
    assert v1[-1] == pytest.approx(.6)
    assert m0 in (0, 1, 2) and m1 in (0, 1, 2)
    # Removing position / HP follows the training summary's explicit coverage
    # and unknown-field conventions; target data is not consulted.
    assert v1[9] == 0.0
    assert v1[11] == 0.0


def test_latest_online_features_equal_offline_batch_features():
    snapshots = [_snapshot(t) for t in (0, 3, 6, 12, 18, 30, 45, 60, 90)]
    rows, modes = [], []
    previous = None
    for snapshot in snapshots:
        row, mode = observable_snapshot_base_features(snapshot, previous)
        rows.append(row)
        modes.append(mode)
        previous = snapshot["issued_at"]
    x = build_mode_change_features(
        np.asarray(rows), np.asarray([s["issued_at"] for s in snapshots]),
        np.asarray(modes), group_ids=[("match-1", "红")] * len(rows))
    latest = build_latest_mode_change_features(
        np.asarray(rows), np.asarray([s["issued_at"] for s in snapshots]),
        np.asarray(modes))
    assert x.shape == (len(snapshots), FEATURE_DIM)
    np.testing.assert_array_equal(latest, x[-1])


def test_latest_features_respect_first_observation_and_history_boundaries():
    # The first row has no prior-window deltas. A new stream must start from
    # scratch, and duplicate issue times are rejected rather than leaked.
    base = np.zeros((1, BASE_FEATURE_DIM), dtype=np.float32)
    x0 = build_latest_mode_change_features(base, np.array([3.]), np.array([1]))
    assert x0.shape == (FEATURE_DIM,)
    assert x0[BASE_FEATURE_DIM + 3] == 1.0
    assert x0[-1] == 0.0
    with pytest.raises(ValueError, match="unique and increasing"):
        build_latest_mode_change_features(
            np.zeros((2, BASE_FEATURE_DIM), dtype=np.float32),
            np.array([3., 3.]), np.array([1, 1]))


def test_valid_mode_xgb_candidate_replaces_only_the_30s_head_and_keeps_baseline():
    horizons = {
        '10': {'global_mode': {'top': '均衡', 'confidence': .7, 'probabilities': {'进攻': .1, '均衡': .7, '防守': .2}}},
        '30': {
            'global_mode': {'top': '均衡', 'confidence': .6, 'probabilities': {'进攻': .2, '均衡': .6, '防守': .2}},
            'opponent_trend': {'top': '推进', 'confidence': .7, 'probabilities': {'推进': .7, '稳定': .2, '回撤': .1}},
            'enemy_region': {'top': '中路', 'confidence': .4, 'probabilities': {'中路': .4}},
            'entities': {},
        },
    }
    candidate = {'status': 'ok', 'horizon_seconds': 30, 'top': '进攻', 'confidence': .61,
                 'probabilities': {'进攻': .61, '均衡': .24, '防守': .15}}
    applied = apply_mode_xgb_candidate(horizons, candidate, team='红')
    assert applied is True
    assert horizons['10']['global_mode']['top'] == '均衡'
    assert horizons['30']['global_mode']['top'] == '进攻'
    assert horizons['30']['global_mode_neural_baseline']['top'] == '均衡'
    assert horizons['30']['global_mode_source'] == 'mode_change_xgb_v1'


def test_auto_resolver_keeps_xgb_as_sidecar_unless_explicitly_opted_in(monkeypatch):
    captured = {}

    class RuntimeStub:
        def __init__(self, *args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs

    monkeypatch.setattr(temporal_runtime, 'TemporalForecastRuntime', RuntimeStub)
    temporal_runtime.resolve_temporal_forecaster('auto', adaptive_router=None)
    assert captured['args'][0] == temporal_runtime.DEFAULT_CHECKPOINT
    assert captured['kwargs']['experimental_mode_xgb_path'] == temporal_runtime.MODE_XGB_CHECKPOINT
    assert captured['kwargs']['apply_experimental_mode_xgb'] is False

    temporal_runtime.resolve_temporal_forecaster(
        'auto', adaptive_router=None, apply_experimental_mode_xgb=True)
    assert captured['kwargs']['apply_experimental_mode_xgb'] is True
