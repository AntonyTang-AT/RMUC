from finetune.lora_train import compact_training_instruction
from finetune.infer_one import extract_decision


def test_compact_training_instruction_preserves_decision_sections_and_bound():
    text = (
        '局势: t=10 | 红: 步兵3在红中路 hp1 | 蓝: 哨兵在蓝中路 hp1 | 时间尺度: 10s | '
        '全局分析: mode=均衡 threat=低 | 现状（己方）: 我方重点：无 | '
        '现状（敌方）: 敌方重点：哨兵接近 | 预测解析: 10s后可能推进 | '
        '整体建议（仅作建议）: 保持观察 | 自然语言态势: 当前双方尚未接触 | '
        '预测参考信息（因果模型输出）：' + ('x' * 3000)
    )
    compact = compact_training_instruction(text, max_chars=980)
    assert len(compact) <= 980
    assert '全局分析:' in compact
    assert '预测解析:' in compact
    assert '整体建议' in compact


def test_compact_training_instruction_preserves_forecast_heads_and_baselines():
    import json

    context = {
        'schema_version': 'forecast-context-v1',
        'predictions': [
            {
                'horizon_seconds': horizon,
                'global_mode': {'top': '均衡', 'confidence': .72},
                'opponent_trend': {'top': '推进', 'confidence': .61},
                'enemy_region': {'top': 'x3y1', 'confidence': .54},
                'model_comparison': {
                    'global_mode': {'model': '均衡', 'baseline': '均衡'},
                    'opponent_trend': {'model': '推进', 'baseline': '稳定'},
                    'enemy_region': {'model': 'x3y1', 'baseline': 'x2y1'},
                },
            }
            for horizon in (10, 30, 60, 120)
        ],
    }
    text = ('局势: t=10 | 红: 我方状态 | 蓝: 敌方状态 | 预测参考信息：\n'
            + json.dumps(context, ensure_ascii=False))

    compact = compact_training_instruction(text, max_chars=980)

    assert len(compact) <= 980
    assert '10s' in compact and '30s' in compact
    assert '区域=x3y1' in compact
    assert 'M=x3y1/P=x2y1' in compact
    assert '趋势=推进' in compact
    assert 'P=x3y1:0.54' not in compact  # this fixture has no distribution payload


def test_compact_training_instruction_preserves_forecast_when_other_sections_are_long():
    import json

    context = {
        'predictions': [{
            'horizon_seconds': horizon,
            'global_mode': {'top': '均衡', 'confidence': .72,
                            'probabilities': {'均衡': .72, '进攻': .20, '防守': .08}},
            'opponent_trend': {'top': '推进', 'confidence': .61,
                               'probabilities': {'推进': .61, '稳定': .39}},
            'enemy_region': {'top': 'x3y1', 'confidence': .54,
                             'probabilities': {'x3y1': .54, 'x2y1': .31}},
            'model_comparison': {
                'enemy_region': {'model': 'x3y1', 'baseline': 'x2y1'},
            },
        } for horizon in (10, 30, 60, 120)],
    }
    text = ('局势: t=10 | 红: ' + ('我方可见事实 ' * 30) +
            ' | 蓝: ' + ('敌方可见事实 ' * 30) +
            ' | 预测解析: ' + ('较长叙述 ' * 30) +
            ' | 预测参考信息（因果模型输出）：' +
            json.dumps(context, ensure_ascii=False))

    compact = compact_training_instruction(text, max_chars=980)

    assert len(compact) <= 980
    assert '预测参考(非事实)' in compact
    assert '10s' in compact and '30s' in compact
    assert '区域=x3y1(0.54;P=x3y1:0.54,x2y1:0.31;M=x3y1/P=x2y1)' in compact

    inference_compact = compact_training_instruction(text)
    assert len(inference_compact) <= 760
    assert '预测参考(非事实)' in inference_compact
    assert '10s' in inference_compact


def test_compact_training_instruction_reads_selected_only_forecast_fallback():
    import json

    context = {'predictions': [
        {'horizon_seconds': 10,
         'selected': {'mode': '进攻', 'trend': '推进', 'region': '蓝中路'}},
    ]}
    text = ('局势: t=10 | 预测参考信息（因果模型输出）：' + json.dumps(
        context, ensure_ascii=False) + ' | 其他内容 ' + ('x' * 1200))

    compact = compact_training_instruction(text, max_chars=980)

    assert '10s' in compact
    assert '模式=进攻(?)' in compact
    assert '趋势=推进(?)' in compact
    assert '区域=蓝中路(?)' in compact


def test_extract_decision_normalizes_enum_whitespace():
    raw = '''{"t":1,"global_mode":"均衡","units":{
      "英雄":{"target":"无","move":"推进","fire":false,"risk":"低"},
      "工程":{"target":"无","move":"推进","fire":false,"risk":"低"},
      "步兵3":{"target":"无","move":"推进","fire":false,"risk":"低"},
      "步兵4":{"target":"无","move":"推进","fire":false,"risk":"低"},
      "空中":{"target":"无","move":" 游击 ","fire":false,"risk":"低"},
      "哨兵":{"target":"无","move":"推进","fire":false,"risk":"低"}}}'''
    decision = extract_decision(raw)
    assert decision is not None
    assert decision['units']['空中']['move'] == '游击'
