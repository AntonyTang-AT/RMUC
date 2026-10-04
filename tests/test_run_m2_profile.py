# -*- coding: utf-8 -*-
"""run_m2.apply_profile 轻量单测（无 torch/GPU）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import run_m2 as m2


def test_mainline_forces_skip_llm_and_injects_lookahead():
    f = m2.apply_profile('mainline', skip_llm=False, lookahead_delta=0, skip_rl=False)
    assert f['profile'] == 'mainline'
    assert f['skip_llm'] is True
    assert f['lookahead_delta'] == m2.MAINLINE_LOOKAHEAD_DELTA
    assert f['skip_rl'] is False


def test_mainline_keeps_explicit_lookahead():
    f = m2.apply_profile('mainline', skip_llm=False, lookahead_delta=8)
    assert f['skip_llm'] is True
    assert f['lookahead_delta'] == 8


def test_research_preserves_flags():
    f = m2.apply_profile('research', skip_llm=False, lookahead_delta=0, skip_rl=True)
    assert f['skip_llm'] is False
    assert f['lookahead_delta'] == 0
    assert f['skip_rl'] is True


def test_unknown_profile_raises():
    try:
        m2.apply_profile('nope', False, 0)
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_formal_tier2_is_default_but_all_fallback_disables_it():
    assert m2.resolve_tier2_adapter('auto') == 'auto'
    assert m2.resolve_tier2_adapter(None) == 'auto'
    assert m2.resolve_tier2_adapter('auto', all_fallback=True) == 'none'


def test_observable_snapshots_load_for_tier2_or_temporal_forecasting():
    assert m2.needs_observable_snapshots('auto', 'none') is True
    assert m2.needs_observable_snapshots('none', 'checkpoint.pt') is True
    assert m2.needs_observable_snapshots('none', 'none') is False


def test_observable_snapshots_align_to_three_second_decision_clock():
    dps = [{'t': 1.0}, {'t': 4.0}, {'t': 7.0}]
    snapshots = [
        {'issued_at': 1.0, 'id': 'a'},
        {'issued_at': 2.0, 'id': 'b'},
        {'issued_at': 4.0, 'id': 'c'},
        {'issued_at': 7.0, 'id': 'd'},
    ]
    aligned = m2.align_observable_snapshots(dps, snapshots)
    assert [row['id'] if row else None for row in aligned] == ['a', 'c', 'd']


def test_precompute_temporal_forecasts_consumes_all_snapshots_in_order():
    dps = [{'t': 1.0}, {'t': 4.0}]
    snapshots = [
        {'issued_at': 4.0}, {'issued_at': 1.0}, {'issued_at': 2.0},
        {'issued_at': 3.0},
    ]

    class FakeForecaster:
        def __init__(self):
            self.seen = []

        def predict(self, row):
            self.seen.append(row['issued_at'])
            return {'issued_at': row['issued_at']}

    forecaster = FakeForecaster()
    outputs = m2.precompute_temporal_forecasts(dps, snapshots, forecaster)
    assert forecaster.seen == [1.0, 2.0, 3.0, 4.0]
    assert outputs == [{'issued_at': 1.0}, {'issued_at': 4.0}]
