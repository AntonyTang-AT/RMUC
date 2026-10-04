from runtime.temporal_forecast import _derive_global_outlook


def _unit(side, kind, zone, **extra):
    return {'side': side, 'entity_type': kind, 'state': 'active',
            'to_zone': zone, 'presence_probability': .8, **extra}


def _outlook(entities, mode='均衡'):
    return _derive_global_outlook(entities,
                                  {'top': mode, 'confidence': .7},
                                  {'top': '持平', 'confidence': .5},
                                  {'top': '中央'}, '红', 30)


def test_contested_future_zone_gets_specific_grounded_tactical_label():
    outlook = _outlook({
        'self:英雄': _unit('self', '英雄', '中央通道'),
        'opponent:英雄': _unit('opponent', '英雄', '中央通道'),
    })
    situation = outlook['tactical_situation']
    assert situation['label'] == '关键区域争夺'
    assert '中央通道' in situation['evidence']
    assert situation['probability'] is None


def test_predicted_enemy_intrusion_prioritizes_defense_pressure():
    outlook = _outlook({
        'self:英雄': _unit('self', '英雄', '红中路'),
        'opponent:英雄': _unit('opponent', '英雄', '红基地'),
    }, mode='防守')
    assert outlook['tactical_situation']['label'] == '纵深拦截压力'
    assert outlook['tactical_situation']['source'] == 'forecast_evidence_rule'


def test_coarse_three_way_mode_alone_does_not_invent_tactics():
    situation = _outlook({}, mode='进攻')['tactical_situation']
    assert situation['label'] == '细分态势待确认'
    assert situation['evidence'] == []


def test_blue_view_detects_own_pressure_toward_red_side():
    outlook = _derive_global_outlook({
        'self:英雄': _unit('self', '英雄', '红前沿'),
        'opponent:英雄': _unit('opponent', '英雄', '红基地'),
    }, {'top': '进攻', 'confidence': .7}, {'top': '持平', 'confidence': .5},
        {'top': '红侧'}, '蓝', 30)
    assert outlook['decision_summary']['control_trend'] == 'own_pressure'
    assert outlook['tactical_situation']['label'] == '控制性推进'
