from eval.temporal_consistency import evaluate_replay_frames


def _frame(t, mode, trend, actual_mode='均衡', actual_trend='稳定', zone='红中路'):
    forecast = {
        'mode': mode, 'mode_probabilities': {'进攻': 0.8 if mode == '进攻' else 0.1,
                                             '均衡': 0.8 if mode == '均衡' else 0.1,
                                             '防守': 0.8 if mode == '防守' else 0.1},
        'trend': trend, 'trend_probabilities': {'推进': 0.8 if trend == '推进' else 0.1,
                                                '稳定': 0.8 if trend == '稳定' else 0.1,
                                                '回撤': 0.8 if trend == '回撤' else 0.1},
        'entities': {'self:英雄': {'to_zone': zone}},
    }
    return {'t': t, 'views': {'红': {'horizons': {
        '10': {'forecast': forecast, 'actual': {'mode': actual_mode, 'trend': actual_trend}, 'entities': forecast['entities']},
        '30': {'forecast': forecast, 'actual': {'mode': actual_mode, 'trend': actual_trend}, 'entities': forecast['entities']},
    }}, '蓝': {'horizons': {
        '10': {'forecast': forecast, 'actual': {'mode': actual_mode, 'trend': actual_trend}, 'entities': forecast['entities']},
        '30': {'forecast': forecast, 'actual': {'mode': actual_mode, 'trend': actual_trend}, 'entities': forecast['entities']},
    }}}}


def test_consistency_uses_nearest_three_second_tick_for_twenty_seconds():
    payload = {
        'decision_step_seconds': 3,
        'frames': [_frame(t, '均衡', '稳定') for t in range(0, 40, 3)],
    }
    result = evaluate_replay_frames(payload)
    assert result['pairs'] > 0
    assert result['offset_used_seconds'] == 21
    assert result['offset_quantization_error_seconds'] == 1
    assert result['by_task']['global_mode']['n'] > 0

