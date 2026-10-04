from eval.compare_lora_forecast_context import _score_decisions


def _decision(move):
    return {
        't': 1.0,
        'global_mode': '均衡',
        'units': {
            name: {'target': '无', 'move': move if name == '哨兵' else '游击',
                   'fire': False, 'risk': '低'}
            for name in ('英雄', '工程', '步兵3', '步兵4', '空中', '哨兵')
        },
    }


def test_score_decisions_keeps_rule_moves_aligned_after_parse_failure():
    result = _score_decisions(
        [_decision('推进'), None, _decision('回防')],
        ['推进', '推进', '回防'],
    )

    assert result['parsed'] == 2
    assert result['agreement_with_rule'] == 1.0
