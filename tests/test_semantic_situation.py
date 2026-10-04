# -*- coding: utf-8 -*-
"""semantic_situation 无 torch 单测。"""
from worldstate import semantic as sem


def _w(enemy_x=0.7, our_x=0.3, base_hp=1.0, enemy_types=('步兵3',)):
    robots = {
        's': {'type': '哨兵', 'team': '红', 'norm_xy': [our_x, 0.5], 'hp_ratio': 1.0},
    }
    for i, t in enumerate(enemy_types):
        robots[f'e{i}'] = {'type': t, 'team': '蓝', 'norm_xy': [enemy_x, 0.5], 'hp_ratio': 1.0}
    return {
        't': 0.0,
        'robots': robots,
        'buildings': {
            'rb': {'type': '基地', 'team': '红', 'hp_ratio': base_hp},
            'ro': {'type': '前哨站', 'team': '红', 'hp_ratio': 1.0},
            'bo': {'type': '前哨站', 'team': '蓝', 'hp_ratio': 1.0},
        },
    }


def test_base_threat_on_low_base():
    s = sem.derive_semantic_situation(_w(base_hp=0.3, enemy_x=0.8))
    assert s['base_threat'] is True
    assert s['source'] == 'replay_approx'


def test_enemy_trend_press():
    prev = _w(enemy_x=0.7)
    cur = _w(enemy_x=0.55)
    s = sem.derive_semantic_situation(cur, prev_w=prev)
    assert s['enemy_trend'] == 'press'


def test_lost_target():
    prev = _w(enemy_types=('步兵3', '英雄'))
    cur = _w(enemy_types=('步兵3',))
    s = sem.derive_semantic_situation(cur, prev_w=prev)
    assert s['lost_target'] is True
    assert '英雄' in s['lost_types']


def test_format_line():
    s = sem.derive_semantic_situation(_w(base_hp=0.2))
    line = sem.format_semantic_line(s)
    assert 'base_threat=1' in line
    assert '语义:' in line


def test_global_analysis_has_evidence_and_horizon_specificity():
    analysis = sem.derive_global_analysis(_w(enemy_x=0.2, our_x=0.8), team='红')
    assert analysis['schema_version'] == 'global-analysis-v1'
    assert analysis['state']['opponent_push_count'] >= 1
    assert analysis['horizon_views']['10s']['specificity'] == 'high'
    assert analysis['horizon_views']['120s']['specificity'] == 'coarse'

def test_global_analysis_reports_functional_zones():
    analysis = sem.derive_global_analysis(_w(enemy_x=0.7, our_x=0.2), team='红')
    assert analysis['state']['own_functional_zones']
    assert analysis['state']['opponent_functional_zones']
    assert 'zone' in analysis['state']['unit_locations'][0]
    assert ' → ' not in analysis['state']['unit_locations'][0]['zone']
    assert analysis['narrative']['description']


def test_unknown_hp_stays_in_presence_counts_but_is_not_labeled_alive():
    robots = {
        'active': {'entity_id': 'active', 'type': '步兵3', 'team': '红',
                   'norm_xy': [.2, .3], 'hp_ratio': .8},
        'dead': {'entity_id': 'dead', 'type': '步兵4', 'team': '红',
                 'norm_xy': [.2, .4], 'hp_ratio': 0.0},
        'missing': {'entity_id': 'missing', 'type': '工程', 'team': '红',
                    'norm_xy': [.25, .5]},
        'nan': {'entity_id': 'nan', 'type': '哨兵', 'team': '红',
                'norm_xy': [.3, .6], 'hp_ratio': float('nan')},
        'bool': {'entity_id': 'bool', 'type': '英雄', 'team': '红',
                 'norm_xy': [.35, .7], 'hp_ratio': True},
    }
    world = {
        't': 1.0,
        'robots': robots,
        'buildings': {
            'base': {'type': '基地', 'team': '红', 'hp_ratio': 1.0},
            'outpost': {'type': '前哨站', 'team': '红', 'hp_ratio': 1.0},
        },
    }

    analysis = sem.derive_global_analysis(world, team='红')
    locations = {row['entity_id']: row for row in analysis['state']['unit_locations']}
    narrative_units = {row['entity_id']: row
                       for row in analysis['narrative']['sections']['unit_analysis']}

    assert locations['active']['life_state'] == 'active'
    assert locations['dead']['life_state'] == 'destroyed'
    assert analysis['state']['own_mean_hp'] == .8  # NaN/bool/missing are not folded into a mean.
    for entity_id in ('missing', 'nan', 'bool'):
        assert locations[entity_id]['life_state'] == 'unknown'
        assert narrative_units[entity_id]['life_state'] == 'unknown'
        assert '存活状态未知，当前仍可观测' in narrative_units[entity_id]['current_status']
    # Unknown HP remains a candidate for the visible roster/zone counts;
    # only the confirmed zero-HP unit is excluded.
    assert sum(analysis['state']['own_functional_zones'].values()) == 4


def test_global_analysis_exposes_control_risk_resource_and_actions():
    world = _w(enemy_x=0.45, our_x=0.55, base_hp=0.4)
    analysis = sem.derive_global_analysis(world, team='红', mode='防守')
    assert analysis['state']['control_posture'] in ('contested', 'opponent_advantage')
    assert 'zone_control' in analysis['state']
    assert analysis['state']['risk_register']['items']
    assert analysis['recommendations']
    assert analysis['decision_gates']
    # Narrative should explain the situation and action logic without dumping
    # the zone count table into prose.
    assert '功能区：' not in analysis['narrative']['description']
    assert '建议：' in analysis['narrative']['recommendation']


def test_global_narrative_keeps_numeric_evidence_out_of_prose():
    from worldstate.narrative import build_global_narrative
    narrative = build_global_narrative({
        'current_mode': '防守', 'threat_level': '高',
        'evidence': ['敌方越过中线单位 3 个，高于我方 1 个'],
        'state': {'team': '红', 'enemy_trend': 'press'},
    })
    assert '3 个' not in narrative['description']
    assert narrative['description'].startswith('现况：')


def test_global_narrative_prose_is_three_scan_friendly_sentences_and_keeps_details():
    from worldstate.narrative import build_global_narrative

    narrative = build_global_narrative({
        'current_mode': '防守', 'confidence': .78,
        'state': {
            'team': '红', 'enemy_trend': 'press', 'own_base_hp': .42,
            'visible_unit_states': [
                {'entity_id': 'r1', 'team': '红', 'type': '哨兵', 'zone': '红方前哨',
                 'life_state': 'active', 'state': {'defense_status': 'guarding_outpost'}},
                {'entity_id': 'b1', 'team': '蓝', 'type': '步兵', 'zone': '红方前哨',
                 'life_state': 'active', 'state': {'attack_status': 'attacking'}},
            ],
            'unit_locations': [
                {'entity_id': 'r1', 'team': '红', 'zone': '红方前哨', 'hp_ratio': .8},
                {'entity_id': 'b1', 'team': '蓝', 'zone': '红方前哨', 'hp_ratio': .8},
            ],
            'zone_control': {'contested_zones': ['红方前哨']},
        },
        'forecast_context': {'predictions': [{
            'horizon_seconds': 10,
            'opponent_trend': {'top': 'press', 'confidence': .78},
            'global_mode': {'top': '防守', 'confidence': .63},
        }]},
        'detailed_posture': {
            'label': '基地防守', 'objective': '守住前哨入口',
            'trigger': '敌方连续两帧向前哨推进', 'fallback': '退回基地侧掩护区',
        },
    })

    brief = narrative['paragraph']
    assert brief.count('。') == 3
    assert brief.startswith('现况：')
    assert '预测解析' in brief and '建议：' in brief
    assert '敌方可能推进' in brief
    assert '红方前哨' in brief and '敌方' in brief
    assert '时执行' in brief
    assert narrative['sections']['overall_recommendation']['trigger'] == '敌方连续两帧向前哨推进'
    # Compression applies only to prose; structured evidence remains available.
    assert narrative['sections']['forecast_analysis']['horizons']
    assert narrative['sections']['unit_analysis']


def test_fine_grained_posture_has_stable_code_and_auditable_evidence():
    from worldstate.semantic import _detailed_posture

    posture = _detailed_posture(
        mode='防守', threat='高', risks={'enemy_intruders': [{'zone': '红基地'}]},
        zone_control={}, special_summary={}, situation={}, own_base=.9,
        own_outpost=.9, control_posture='contested')
    assert posture['code'] == 'deep_interception'
    assert posture['label'] == '纵深拦截'
    assert posture['evidence_conditions'] == ['发现敌方单位进入己方防区']
    assert posture['source'] == 'evidence_rule_derived_posture'


def test_unit_advice_keeps_observation_action_reason_and_stop_rule_together():
    from worldstate.narrative import build_global_narrative

    narrative = build_global_narrative({
        'current_mode': '均衡',
        'state': {
            'team': '红',
            'visible_unit_states': [{
                'entity_id': 'r-low', 'team': '红', 'type': '步兵3', 'zone': '红中路',
                'state': {'decision_guidance': {
                    'applicable': True,
                    'current_state': '血量偏低（20%），位于红中路',
                    'action': '撤向红侧掩护区',
                    'first_step_zone': '红侧掩护区',
                    'target_zone': '红侧掩护区',
                    'trigger_condition': '当前HP比例低于0.30',
                    'cancel_condition': 'HP恢复到0.50以上或路线受阻',
                }},
            }],
            'unit_locations': [{'entity_id': 'r-low', 'team': '红', 'zone': '红中路', 'hp_ratio': .2}],
        },
    })

    unit = narrative['sections']['unit_analysis'][0]
    assert unit['observed_state'] == '血量偏低（20%），位于红中路'
    assert unit['recommendation'] == '撤向红侧掩护区'
    assert unit['reason'] == '依据当前观测：血量偏低（20%），位于红中路'
    assert unit['first_step_zone'] == '红侧掩护区'
    assert unit['trigger_condition'] == '当前HP比例低于0.30'
    assert unit['cancel_condition'] == 'HP恢复到0.50以上或路线受阻'
    assert unit['advice_text'] == '现在：撤向红侧掩护区；触发：当前HP比例低于0.30。'
    assert unit['display_advice']['stop_when'] == 'HP恢复到0.50以上或路线受阻'


def test_unit_without_action_evidence_is_not_rendered_as_a_hold_command():
    from worldstate.narrative import build_global_narrative

    narrative = build_global_narrative({
        'current_mode': '均衡',
        'state': {
            'team': '红',
            'visible_unit_states': [{
                'entity_id': 'r1', 'team': '红', 'type': '步兵3', 'zone': '红中路',
                'state': {'decision_guidance': {
                    'applicable': False, 'action_label': 'no_action_evidence',
                    'current_state': '位于红中路，任务/交战状态尚未确认',
                    'action': None,
                    'short_text': '暂无足够证据支持具体移动或攻击；先确认任务或可见目标，再重新评估。',
                    'evaluation': {'scorable': False},
                }},
            }],
            'unit_locations': [{'entity_id': 'r1', 'team': '红', 'zone': '红中路'}],
        },
    })
    unit = narrative['sections']['unit_analysis'][0]
    assert unit['recommendation'] is None
    assert unit['recommendation_type'] == 'insufficient_action_evidence'
    assert '现在：暂不建议具体移动或攻击' in unit['advice_text']
    assert '保持当前区域' not in unit['advice_text']


def test_global_brief_uses_three_plain_language_sections():
    from worldstate.narrative import build_global_narrative

    narrative = build_global_narrative({
        'current_mode': '防守',
        'state': {'team': '红', 'enemy_trend': 'press'},
        'forecast_context': {'predictions': [{
            'horizon_seconds': 10,
            'opponent_trend': {'top': 'press', 'confidence': .78},
        }]},
        'detailed_posture': {
            'objective': '守住基地入口',
            'trigger': '连续观测到敌方进入防区',
        },
    })

    lines = narrative['paragraph'].splitlines()
    assert len(lines) == 3
    assert lines[0].startswith('现况：')
    assert lines[1].startswith('预测解析')
    assert lines[2].startswith('建议：')


def test_global_narrative_separates_observed_events_enemy_forecast_and_own_advice():
    from worldstate.narrative import build_global_narrative

    narrative = build_global_narrative({
        'current_mode': '均衡',
        'state': {
            'team': '红', 'enemy_trend': 'hold',
            'visible_unit_states': [
                {'entity_id': 'r1', 'team': '红', 'type': '步兵', 'zone': '红中路',
                 'life_state': 'active', 'state': {'attack_status': 'attacking'}},
                {'entity_id': 'b1', 'team': '蓝', 'type': '步兵', 'zone': '蓝中路',
                 'life_state': 'active', 'state': {'attack_status': 'possible_attack'}},
            ],
            'unit_locations': [
                {'entity_id': 'r1', 'team': '红', 'type': '步兵', 'zone': '红中路', 'hp_ratio': .8},
                {'entity_id': 'b1', 'team': '蓝', 'type': '步兵', 'zone': '蓝中路', 'hp_ratio': .8},
            ],
        },
        'forecast_context': {'predictions': [{
            'horizon_seconds': 10,
            'entities': [{'side': 'opponent', 'entity_type': '步兵', 'entity_id': 'b1',
                          'from_zone': '蓝中路', 'to_zone': '中央高地'}],
        }]},
    })

    current = narrative['sections']['current_situation']
    units = narrative['sections']['unit_analysis']
    enemy = next(unit for unit in units if unit['relation'] == '敌方')
    own = next(unit for unit in units if unit['relation'] == '己方')
    assert current['special_events']
    assert narrative['description'].startswith('现况：接敌交战')
    assert enemy['analysis_role'] == 'enemy_forecast'
    assert '蓝中路到中央高地' in enemy['future_prediction']
    assert enemy['recommendation'] is None
    assert own['analysis_role'] == 'friendly_recommendation'
    # An attack-status flag alone does not identify a target or support a
    # concrete unit action; do not fall back to generic attack advice.
    assert own['recommendation_type'] == 'insufficient_action_evidence'
    assert own['recommendation'] is None
    assert own['future_prediction'] is None
