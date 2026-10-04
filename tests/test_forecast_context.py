import json

from worldstate.forecast_context import build_forecast_context, format_forecast_context
from finetune.make_dataset import situation_digest


def _forecast():
    return {
        'schema_version': 'temporal-entity-forecast-v1',
        'model_version': 'hybrid-v1',
        'calibration_version': 'cal-v1',
        'input_snapshot_hash': 'abc123',
        'team': '红', 'issued_at': 12.0, 'causal_history_length': 40,
        'horizons': {
            '10': {
                'global_mode': {'top': '进攻', 'confidence': .8,
                                'probabilities': {'进攻': .8, '均衡': .15, '防守': .05}},
                'opponent_trend': {'top': '推进', 'confidence': .7,
                                   'probabilities': {'推进': .7, '稳定': .3}},
                'enemy_region': {'top': '蓝中路', 'confidence': .6,
                                 'probabilities': {'蓝中路': .6}},
                'entities': {'opponent:英雄': {
                    'entity_id': 'opponent:英雄', 'side': 'opponent',
                    'entity_type': '英雄', 'state': 'active',
                    'presence_probability': .9,
                    'movement': {'from_zone': '蓝前哨区', 'to_zone': '中央通道',
                                 'reachable_within_horizon': True,
                                 'estimated_travel_time_s': 8.0},
                    'special_state_forecast': {
                        'attack_status': 'possible_attack',
                        'possible_target': '己方前哨站',
                    },
                }},
                'global_outlook': {
                    'uncertainty': {'joint_confidence': .56},
                    'control': {'contested_zones': ['中央通道']},
                },
            },
        },
    }


def test_forecast_context_is_versioned_and_reference_only():
    context = build_forecast_context(_forecast())
    assert context['schema_version'] == 'forecast-context-v1'
    assert context['source']['input_snapshot_hash'] == 'abc123'
    prediction = context['predictions'][0]
    assert prediction['horizon_seconds'] == 10
    assert prediction['entities'][0]['from_zone'] == '蓝前哨区'
    assert context['usage_policy']['role'] == 'reference_only'
    assert 'must_not_override' in context['usage_policy']


def test_context_prompt_has_boundary_and_preserves_probability():
    context = build_forecast_context(_forecast())
    text = format_forecast_context(context)
    assert '仅供态势分析' in text
    assert '0.8' in text
    assert '预测参考信息' in situation_digest(
        {'t': 0, 'robots': {}, 'buildings': {}}, forecast_context=context)


def test_context_is_json_serializable():
    context = build_forecast_context(_forecast())
    json.dumps(context, ensure_ascii=False)


def test_context_prompt_is_bounded_for_dense_forecasts():
    context = build_forecast_context(_forecast())
    text = format_forecast_context(context, max_chars=1000)
    assert len(text) <= 1000 + len('预测参考信息（因果模型输出，仅供态势分析；不是事实，也不能直接改写动作）：\n')
    assert 'reference_only' in text
