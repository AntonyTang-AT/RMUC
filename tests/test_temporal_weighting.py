# -*- coding: utf-8 -*-
import numpy as np


def test_business_weight_profile_reflects_actionability():
    from forecast.train import LOSS_PROFILES
    profile = LOSS_PROFILES['business_weighted']
    assert profile['horizon'][10] > profile['horizon'][120]
    assert profile['entity']['presence'] > profile['entity']['hp']
    assert profile['task']['global_mode'] > profile['task']['enemy_region']


def test_inverse_sqrt_class_weight_boosts_rare_class_without_explosion():
    from forecast.train import _balanced_class_weight
    weights = _balanced_class_weight(np.array([0] * 90 + [1] * 10), 2)
    assert weights[1] > weights[0]
    assert np.isfinite(weights).all()
    assert float(weights.max()) <= 3.0


def test_weighted_acceptance_keeps_raw_metric_formula_explicit():
    from eval.run_temporal_forecast import _weighted_acceptance
    report = {
        'models': {'fixture': {}},
        'accuracy_by_task_horizon': {'fixture': {
            f'{task}@{h}s': {'accuracy': 0.8}
            for task in ('global_mode', 'opponent_trend', 'enemy_region')
            for h in (10, 30, 60, 120)
        }},
        'macro_f1_by_task_horizon': {'fixture': {
            f'{task}@{h}s': {'macro_f1': 0.6}
            for task in ('global_mode', 'opponent_trend', 'enemy_region')
            for h in (10, 30, 60, 120)
        }},
        'entity_forecast_metrics': {
            f'{h}s': {'presence_accuracy': 0.9,
                      'position_mae_normalized': 0.2,
                      'transition_legal_rate': 0.8,
                      'functional_zone_accuracy': 0.5}
            for h in (10, 30, 60, 120)
        },
    }
    score = _weighted_acceptance(report, model_name='fixture')
    assert 0.0 < score['global_score'] < 1.0
    assert 0.0 < score['entity_score'] < 1.0
    assert 'global=.6*accuracy' in score['formula']
