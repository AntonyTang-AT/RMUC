import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tree import global_mode as gm
from schema import layered as sl

def _ws(red_x, blue_x, red_hp, blue_hp, base_hp):
    # 红 3 台 + 蓝 3 台; 中线 x=0.5 (norm)
    def u(tid, team, x, hp):
        return {'type': tid, 'team': team, 'zone': 'Z01', 'norm_xy': [x, 0.5],
                'hp_ratio': hp, 'power': 0.0, 'heat': 0.0, 'vuln': 0,
                'ammo17': 0, 'ammo42': 0, 'dfire': 0.0}
    reds = {i: u('哨兵', '红', x, hp) for i, (x, hp) in enumerate(zip(red_x, red_hp))}
    blues = {i + 10: u('步兵3', '蓝', x, hp) for i, (x, hp) in enumerate(zip(blue_x, blue_hp))}
    return {'t': 0.0, 'team': {}, 'robots': {**reds, **blues},
            'buildings': {99: {'type': '基地', 'team': '红', 'hp_ratio': base_hp}}}

def test_three_modes_observable():
    # 进攻: 红 units 深入敌方半场(x>0.5) 且蓝残血(蓝缩在自己半场)
    assert gm.derive_global_mode(_ws([0.8, 0.7, 0.6], [0.7, 0.65, 0.6], [1,1,1], [0.1,0.1,0.1], 1.0)) == '进攻'
    # 防守: 蓝 units 深入红方半场(x<0.5) 且红残血/基地低血
    assert gm.derive_global_mode(_ws([0.4, 0.35, 0.3], [0.45, 0.4, 0.35], [0.1,0.1,0.1], [1,1,1], 0.4)) == '防守'
    # 均衡: 中线拉扯, hp 均衡
    assert gm.derive_global_mode(_ws([0.6, 0.5, 0.4], [0.4, 0.5, 0.6], [0.7,0.7,0.7], [0.7,0.7,0.7], 0.9)) == '均衡'

def test_unit_intent_channels_valid():
    w = _ws([0.8, 0.7, 0.6], [0.2, 0.2, 0.2], [1,1,1], [0.1,0.1,0.1], 1.0)
    mode = gm.derive_global_mode(w)
    unit = w['robots'][0]
    it = gm.derive_unit_intent(unit, w, mode)
    assert it['target'] in sl.TARGETS
    assert it['move'] in sl.MOVES
    assert it['risk'] in sl.RISKS
    assert isinstance(it['fire'], bool)

def test_base_threat_ignores_enemy_team():
    # 敌方(蓝)基地低血但局势均衡 -> 红视角不可判防守; 红基地完好
    ws = _ws([0.6, 0.5, 0.4], [0.4, 0.5, 0.6], [0.7, 0.7, 0.7], [0.7, 0.7, 0.7], 1.0)
    ws['buildings'] = {99: {'type': '基地', 'team': '红', 'hp_ratio': 1.0},
                       200: {'type': '基地', 'team': '蓝', 'hp_ratio': 0.2}}
    assert gm.derive_global_mode(ws) != '防守'

def test_base_threat_overrides_attack_push():
    # 红深入压蓝(进攻条件满足)且红基地低血 -> 防守必须压倒进攻
    assert gm.derive_global_mode(_ws([0.8, 0.7, 0.6], [0.7, 0.65, 0.6], [1, 1, 1], [0.1, 0.1, 0.1], 0.4)) == '防守'

def test_derive_sentry_state_three_way():
    """瞬时几何：def / hitbuilding / cruise 三态齐全（对齐考核原图）。"""
    # def: 蓝方单位进入红半场 (nx<0.5)
    w_def = {
        'robots': {
            's': {'type': '哨兵', 'team': '红', 'norm_xy': [0.3, 0.5], 'hp_ratio': 1.0,
                  'vuln': 0, 'dfire': 0},
            'e': {'type': '步兵3', 'team': '蓝', 'norm_xy': [0.3, 0.5], 'hp_ratio': 1.0,
                  'vuln': 0, 'dfire': 0},
        },
        'buildings': {},
    }
    assert gm.derive_sentry_state(w_def) == 'def'
    assert gm.sentry_to_strategic('def') == 'StrategicDefendBase'

    # hitbuilding: 无敌压境，红哨兵越线+开火
    w_hit = {
        'robots': {
            's': {'type': '哨兵', 'team': '红', 'norm_xy': [0.7, 0.5], 'hp_ratio': 1.0,
                  'vuln': 0, 'dfire': 2.0},
            'e': {'type': '步兵3', 'team': '蓝', 'norm_xy': [0.8, 0.5], 'hp_ratio': 1.0,
                  'vuln': 0, 'dfire': 0},
        },
        'buildings': {},
    }
    assert gm.derive_sentry_state(w_hit) == 'hitbuilding'
    assert gm.sentry_to_strategic('hitbuilding') == 'StrategicAttackOutpost'

    # cruise: 中线对峙、未开火
    w_cruise = {
        'robots': {
            's': {'type': '哨兵', 'team': '红', 'norm_xy': [0.4, 0.5], 'hp_ratio': 1.0,
                  'vuln': 0, 'dfire': 0},
            'e': {'type': '步兵3', 'team': '蓝', 'norm_xy': [0.6, 0.5], 'hp_ratio': 1.0,
                  'vuln': 0, 'dfire': 0},
        },
        'buildings': {},
    }
    assert gm.derive_sentry_state(w_cruise) == 'cruise'
    assert gm.sentry_to_strategic('cruise') == 'StrategicPatrol'
    assert gm.derive_sentry_state({'robots': {}, 'buildings': {}}) == 'cruise'
    assert gm.strategic_to_sentry('StrategicDefendBase') == 'def'
