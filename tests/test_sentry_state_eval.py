import pytest


def test_sentry_state_metrics_include_per_class_and_macro_scores():
    from eval.sentry_state_eval import score_state_series

    result = score_state_series(
        ['def', 'hitbuilding', 'cruise', 'def'],
        ['def', 'cruise', 'cruise', 'hitbuilding'],
    )

    assert result['n'] == 4
    assert result['accuracy'] == pytest.approx(0.5)
    assert set(result['per_class']) == {'def', 'hitbuilding', 'cruise'}
    assert result['per_class']['def']['recall'] == pytest.approx(0.5)
    assert 0.0 <= result['macro_f1'] <= 1.0
    assert result['confusion']['def']['def'] == 1
    assert result['confusion']['def']['hitbuilding'] == 1


def test_sentry_state_metrics_reject_mismatched_series():
    from eval.sentry_state_eval import score_state_series

    with pytest.raises(ValueError, match='same length'):
        score_state_series(['def'], ['def', 'cruise'])


def test_global_narrative_contains_description_and_recommendation():
    from worldstate.narrative import build_global_narrative

    analysis = {
        'current_mode': '防守',
        'threat_level': '高',
        'confidence': 0.82,
        'evidence': ['敌方越过中线单位 3 个，高于我方 1 个', '我方建筑受损'],
        'recommended_focus': '收缩阵型，优先处理敌方前排和入侵单位',
        'state': {
            'team': '红', 'time_s': 120.0, 'push_margin': -2,
            'own_base_hp': 0.42, 'opponent_base_hp': 0.95,
            'enemy_trend': 'press',
        },
        'sentry_state': 'def',
    }

    narrative = build_global_narrative(analysis)

    assert narrative['schema_version'] == 'global-narrative-v2'
    assert narrative['description']
    assert narrative['recommendation']
    assert narrative['description'].startswith('现况：')
    assert narrative['sections']['current_situation']['tactical_stage']['label']
    assert '收缩阵型' in narrative['recommendation']
    assert narrative['source'] == 'structured_evidence_template'
