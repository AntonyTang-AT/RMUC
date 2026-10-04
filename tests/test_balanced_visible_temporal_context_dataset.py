from tree.global_mode import derive_unit_intent


def test_blue_attack_intent_targets_enemy_half_symmetrically():
    world = {'robots': {}, 'buildings': {}}
    blue_in_red_half = {'norm_xy': [0.25, 0.5], 'hp_ratio': 1.0, 'dfire': 0.0}
    red_in_blue_half = {'norm_xy': [0.75, 0.5], 'hp_ratio': 1.0, 'dfire': 0.0}

    assert derive_unit_intent(blue_in_red_half, world, '进攻', team='蓝')['target'] == '敌方前哨站'
    assert derive_unit_intent(red_in_blue_half, world, '进攻', team='红')['target'] == '敌方前哨站'


def test_blue_attack_does_not_treat_own_half_as_enemy_half():
    world = {'robots': {}, 'buildings': {}}
    blue_in_own_half = {'norm_xy': [0.75, 0.5], 'hp_ratio': 1.0, 'dfire': 0.0}
    assert derive_unit_intent(blue_in_own_half, world, '进攻', team='蓝')['target'] == '无'
