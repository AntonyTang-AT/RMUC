import json
from pathlib import Path

from worldstate.narrative import build_global_narrative


def _minimal_analysis(units, **state_overrides):
    state = {
        'team': '红', 'visible_unit_states': units, 'unit_locations': [],
        'zone_control': {'contested_zones': []}, 'risk_register': {},
        'enemy_trend': 'unknown',
    }
    state.update(state_overrides)
    return {'current_mode': '均衡', 'state': state, 'recommendations': []}


def test_tactical_stage_uses_observed_events_instead_of_three_way_mode():
    analysis = _minimal_analysis([
        {'entity_id': 'r1', 'team': '红', 'type': '步兵3', 'zone': '红中路',
         'state': {'attack_status': 'attacking',
                   'attacking_target': {'entity_id': 'b1'}}},
        {'entity_id': 'r2', 'team': '红', 'type': '步兵4', 'zone': '红中路',
         'state': {'attack_status': 'attacking',
                   'attacking_target': {'entity_id': 'b1'}}},
        {'entity_id': 'b1', 'team': '蓝', 'type': '英雄', 'zone': '蓝前哨区',
         'state': {'attack_status': 'unknown'}},
    ], zone_control={'contested_zones': ['中央高地']})

    narrative = build_global_narrative(analysis)
    stage = narrative['sections']['current_situation']['tactical_stage']

    assert stage['label'] == '集火压制'
    assert stage['basis'] == 'multiple_friendly_units_confirmed_same_target'
    assert '当前观察' in stage['evidence']
    assert '均衡' not in narrative['description']  # the prose names the observed tactical stage
    assert '争夺中央高地' in narrative['description']


def test_global_stage_does_not_claim_attrition_without_hp_change_evidence():
    narrative = build_global_narrative(_minimal_analysis([
        {'entity_id': 'r1', 'team': '红', 'type': '步兵3', 'zone': '红中路',
         'state': {'attack_status': 'unknown'}},
    ]))

    stage = narrative['sections']['current_situation']['tactical_stage']
    assert stage['label'] == '态势观察'
    assert stage['basis'] == 'no_specific_tactical_event_confirmed'
    assert stage['confidence'] == 'low'


def test_tactical_stage_refines_coarse_mode_with_observed_resource_and_defense_roles():
    resource = build_global_narrative(_minimal_analysis([
        {'entity_id': 'r1', 'team': '红', 'type': '工程', 'zone': '红资源区',
         'state': {'resource_phase': 'returning_to_exchange'}},
    ]))
    guard = build_global_narrative(_minimal_analysis([
        {'entity_id': 'r2', 'team': '红', 'type': '哨兵', 'zone': '红基地',
         'state': {'defense_status': 'guarding_base'}},
    ]))

    assert resource['sections']['current_situation']['tactical_stage']['label'] == '资源回收'
    assert guard['sections']['current_situation']['tactical_stage']['label'] == '基地守卫'
    assert guard['description'].startswith('现况：基地守卫')
    assert '我方' in guard['description'] and '敌方' in guard['description']


def test_tactical_stage_uses_possible_contact_without_claiming_confirmed_attack():
    narrative = build_global_narrative(_minimal_analysis([
        {'entity_id': 'r1', 'team': '红', 'type': '步兵3', 'zone': '中央高地',
         'state': {'attack_status': 'possible_attack'}},
    ]))

    stage = narrative['sections']['current_situation']['tactical_stage']
    assert stage['label'] == '疑似接敌'
    assert stage['confidence'] == 'low'
    assert '可能' in stage['evidence']
    assert '进攻' not in narrative['description']


def test_critical_facility_health_selects_emergency_posture_stage():
    narrative = build_global_narrative(_minimal_analysis(
        [], own_base_hp=.2, own_outpost_hp=.8))

    stage = narrative['sections']['current_situation']['tactical_stage']
    assert stage['label'] == '基地应急防守'
    assert stage['basis'] == 'observed_own_base_hp'
    assert stage['confidence'] == 'high'


def test_single_unit_analysis_separates_friendly_advice_from_enemy_forecast():
    analysis = _minimal_analysis([
        {'entity_id': 'r1', 'team': '红', 'type': '哨兵', 'zone': '红前哨区',
         'hp_ratio': .8,
         'state': {'attack_status': 'unknown', 'decision_guidance': {
             'current_state': '哨兵位于红前哨区；防守命令未知',
             'applicable': True, 'action_label': 'hold_outpost',
             'action': '守住前哨入口', 'short_text': '守住前哨入口',
             'trigger_condition': '确认敌方进入前哨区',
             'cancel_condition': '前哨区威胁解除'}}},
        {'entity_id': 'b1', 'team': '蓝', 'type': '英雄', 'zone': '蓝中路',
         'state': {'attack_status': 'unknown'}},
        {'entity_id': 'b2', 'team': '蓝', 'type': '英雄', 'zone': '蓝前哨区',
         'state': {'attack_status': 'unknown'}},
    ])
    analysis['forecast_context'] = {'predictions': [{
        'horizon_seconds': 10,
        'entities': [{'side': 'opponent', 'entity_id': 'b1',
                      'entity_type': '英雄', 'from_zone': '蓝中路',
                      'to_zone': '中央高地', 'presence': 0.71}],
    }]}

    units = build_global_narrative(analysis)['sections']['unit_analysis']
    friendly = next(unit for unit in units if unit['relation'] == '己方')
    enemy = next(unit for unit in units if unit['relation'] == '敌方')
    other_same_type_enemy = next(unit for unit in units
                                 if unit.get('entity_id') == 'b2')

    assert friendly['current_status'] == '哨兵位于红前哨区；防守命令未知'
    assert friendly['recommended_action'] == '守住前哨入口'
    assert friendly['recommendation_reason'].startswith('依据当前观测：')
    assert friendly['analysis_role'] == 'friendly_recommendation'
    assert enemy['current_status'] == enemy['current_state']
    assert enemy['future_state_predictions'] == [{
        'horizon_seconds': 10, 'from_zone': '蓝中路',
        'to_zone': '中央高地', 'presence_probability': 0.71,
        'status': 'candidate_route',
    }]
    assert enemy['analysis_role'] == 'enemy_forecast'
    assert enemy['recommended_action'] is None
    assert enemy['recommendation_reason'] is None
    assert other_same_type_enemy['future_state_predictions'] == []
    assert '暂无可靠的单体路线预测' in other_same_type_enemy['future_prediction']


def test_real_global_example_has_short_brief_and_keeps_structured_evidence():
    root = Path(__file__).resolve().parents[1]
    sample_path = root / 'out_m2' / 'global_forecast_example_1780143788827_t100.json'
    sample = json.loads(sample_path.read_text(encoding='utf-8'))

    narrative = build_global_narrative(sample['global_analysis'])

    # The visualization keeps both sides, a concrete area/event, forecast
    # uncertainty and an action trigger in a compact ~60-90 character brief.
    assert '我方步兵3交战' in narrative['description']
    assert '敌方英雄交战' in narrative['description']
    assert '中央高地' in narrative['description']
    assert 60 <= len(narrative['paragraph']) <= 90

    assert narrative['forecast_analysis'].startswith('预测解析：')
    assert '方向与置信度未知' in narrative['forecast_analysis']
    assert narrative['recommendation'].startswith('建议：稳住争夺区；')
    assert '确认目标后再投入' in narrative['recommendation']
    assert narrative['paragraph'] == '\n'.join((
        narrative['description'], narrative['forecast_analysis'], narrative['recommendation']))

    # Only the display prose is compacted; original structured facts and the
    # detailed recommendation remain available to the expandable evidence UI.
    assert set(narrative['sections']) == {
        'current_situation', 'forecast_analysis', 'overall_recommendation', 'unit_analysis'}
    assert 'horizons' in narrative['sections']['forecast_analysis']
    assert 'unit_predictions' in narrative['sections']['forecast_analysis']
    facts = narrative['sections']['current_situation']
    assert '己方步兵3在场地开放区正在攻击' in facts['special_events']
    assert '敌我单位出现在同一争夺区域：中央高地、场地开放区' in facts['special_events']
    assert '英雄在场地开放区正在攻击' in facts['opponent_side']
    assert '当争夺区出现可确认的攻击目标时执行' in narrative['sections']['overall_recommendation']['text']
    assert any(row['entity_type'] == '英雄' and row['zone'] == '场地开放区'
               and '正在攻击' in row['current_state']
               for row in narrative['sections']['unit_analysis'])


def test_multi_role_unit_copy_is_direct_and_keeps_enemy_forecast_uncertain():
    analysis = _minimal_analysis([
        {'entity_id': 'r1', 'team': '红', 'type': '工程', 'zone': '红中路',
         'state': {'resource_phase': 'returning_to_exchange', 'decision_guidance': {
             'applicable': True, 'action_label': 'return_resource_to_supply',
             'current_state': '已确认携带资源，位于红中路',
             'action': '沿可通行路线返回红补给区兑换资源',
             'action_reason': '已确认携带资源；红补给区为目的地。',
             'first_step_zone': '红资源通道',
             'trigger_condition': '持有资源且未兑换',
             'cancel_condition': '兑换完成或路线受阻'}}},
        {'entity_id': 'r2', 'team': '红', 'type': '步兵3', 'zone': '红中路',
         'state': {'attack_status': 'possible_attack', 'decision_guidance': {
             'applicable': True, 'action_label': 'cover_and_confirm_target',
             'current_state': '附近约2米有英雄，仅是可能目标',
             'action': '先留在当前区域确认目标，不单独追击',
             'action_reason': '只有距离推断，未确认目标。',
             'trigger_condition': '目标明确可见',
             'cancel_condition': '目标离开视野或确认无目标'}}},
        {'entity_id': 'b1', 'team': '蓝', 'type': '英雄', 'zone': '蓝中路',
         'state': {'attack_status': 'possible_attack'}},
    ])
    analysis['forecast_context'] = {'predictions': [{
        'horizon_seconds': 10,
        'entities': [{'side': 'opponent', 'entity_type': '英雄', 'entity_id': 'b1',
                      'from_zone': '蓝中路', 'to_zone': '中央高地', 'presence': .7}],
    }]}

    units = build_global_narrative(analysis)['sections']['unit_analysis']
    friendly = {unit['entity_id']: unit for unit in units if unit['relation'] == '己方'}
    enemy = next(unit for unit in units if unit['relation'] == '敌方')

    for unit in friendly.values():
        assert unit['display_advice']['action']
        assert unit['display_advice']['reason']
        assert unit['display_advice']['reassess_when']
        assert unit['display_advice']['trigger_when']
        assert unit['display_advice']['stop_when']
        assert '现在：' in unit['advice_text']
        assert '触发：' in unit['advice_text']
    assert '仅是候选路线' in enemy['future_prediction']
    assert '到时核对' in enemy['future_prediction']


def test_compact_global_sides_include_the_relevant_functional_zone():
    narrative = build_global_narrative(_minimal_analysis([
        {'entity_id': 'r1', 'team': '红', 'type': '步兵3', 'zone': '中央高地',
         'state': {'attack_status': 'attacking'}},
        {'entity_id': 'b1', 'team': '蓝', 'type': '英雄', 'zone': '红前哨区',
         'state': {'attack_status': 'possible_attack'}},
    ]))

    assert '步兵3交战（中央高地）' in narrative['description']
    assert '英雄疑似接敌（红前哨区）' in narrative['description']


def test_overall_recommendation_exposes_strategy_tactics_and_trigger_layers():
    narrative = build_global_narrative({
        'current_mode': '防守',
        'state': {'team': '红'},
        'detailed_posture': {
            'code': 'deep_interception', 'label': '纵深拦截',
            'objective': '在基地入口拦截入侵单位',
            'evidence_conditions': ['发现敌方进入己方防区'],
            'source': 'evidence_rule_derived_posture',
            'trigger': '敌方连续两帧向基地推进',
            'fallback': '退回最近己方可控区',
        },
        'force_balance_guidance': {
            'attack': '压住入侵路线', 'defense': '覆盖基地入口', 'reserve': '留一台单位支援',
        },
    })
    layers = narrative['sections']['overall_recommendation']['layers']

    assert layers['strategic']['posture'] == '纵深拦截'
    assert layers['strategic']['evidence'] == ['发现敌方进入己方防区']
    assert layers['tactical'] == {
        'attack': '压住入侵路线', 'defense': '覆盖基地入口', 'reserve': '留一台单位支援',
    }
    assert layers['immediate'] == {
        'trigger': '敌方连续两帧向基地推进', 'fallback': '退回最近己方可控区',
    }


def test_visible_unit_advice_uses_action_specific_reason():
    narrative = build_global_narrative(_minimal_analysis([
        {'entity_id': 'r1', 'team': '红', 'type': '步兵3', 'zone': '红中路',
         'hp_ratio': .8,
         'state': {'decision_guidance': {
             'applicable': True, 'action_label': 'cover_and_confirm_target',
             'current_state': '附近有可能目标，未确认开火目标',
             'action': '留在当前区核验目标',
             'action_reason': '尚未确认弹药指向，原地核验可避免把距离推断当成锁定目标。',
             'trigger_condition': '出现明确开火记录',
             'cancel_condition': '传感器确认没有目标',
         }}}
    ]))
    unit = narrative['sections']['unit_analysis'][0]

    assert unit['reason'].startswith('依据当前观测：附近有可能目标')
    assert '原地核验可避免把距离推断当成锁定目标' in unit['reason']
    assert unit['display_advice']['reason'] == '只有距离推断，目标未确认'
    assert '原地核验可避免把距离推断当成锁定目标' in unit['recommendation_reason']


def test_final_unit_advice_keeps_exact_first_step_and_subject_key_visible():
    narrative = build_global_narrative(_minimal_analysis([
        {'entity_id': 'R3', 'team': '红', 'type': '工程', 'zone': '红中路',
         'state': {'decision_guidance': {
             'unit_ref': {'entity_id': 'R3', 'team': '红', 'entity_type': '工程'},
             'applicable': True, 'action_label': 'return_resource_to_supply',
             'current_state': '已确认携带资源，位于红中路',
             'action': '沿已知路线返回红补给区兑换资源',
             'action_reason': '已确认携带资源；红补给区是明确目的地。',
             'first_step_zone': '红资源通道',
             'movement_goal_zone': '红补给区',
             'target_zone': '红补给区',
             'first_step_distance_m': 2.4,
             'trigger_condition': '持有资源且尚未兑换',
             'cancel_condition': '兑换完成或首步路线受阻',
         }}}
    ]))
    unit = narrative['sections']['unit_analysis'][0]

    assert unit['unit_ref'] == {'entity_id': 'R3', 'team': '红', 'entity_type': '工程'}
    assert unit['recommended_action'] == '沿已知路线返回红补给区兑换资源'
    assert '首步到红资源通道' in unit['display_advice']['action']
    assert '红补给区' in unit['display_advice']['action']
    assert '2.4米' in unit['advice_text']
    assert unit['display_advice']['stop_when'] == '兑换完成或首步路线受阻'


def test_missing_structured_guidance_does_not_fall_back_to_generic_hold_command():
    unit = build_global_narrative(_minimal_analysis([
        {'entity_id': 'R4', 'team': '红', 'type': '步兵3', 'zone': '红中路',
         'state': {'attack_status': 'unknown'}}
    ]))['sections']['unit_analysis'][0]

    assert unit['unit_ref']['entity_id'] == 'R4'
    assert unit['recommended_action'] is None
    assert unit['recommendation_type'] == 'insufficient_action_evidence'
    assert '红中路' in unit['advice_text']
    assert '暂不建议移动或攻击' in unit['advice_text']
    assert '保持相邻区域控制' not in unit['advice_text']


def test_global_and_unit_analysis_keep_enemy_prediction_and_friendly_advice_distinct():
    analysis = _minimal_analysis([
        {'entity_id': 'R3', 'team': '红', 'type': '步兵3', 'zone': '中央高地',
         'hp_ratio': .74,
         'state': {'attack_status': 'attacking', 'decision_guidance': {
             'unit_ref': {'entity_id': 'R3', 'team': '红', 'entity_type': '步兵3'},
             'applicable': True, 'action_label': 'cover_and_confirm_target',
             'current_state': '步兵3在中央高地，附近有疑似目标，未确认攻击对象',
             'action': '留在中央高地掩护，不单独追击',
             'action_reason': '目标只是距离推断，尚无传感器指定对象。',
             'trigger_condition': '确认目标可见且有明确交战信号',
             'cancel_condition': '目标离开视野或本机血量低于30%',
         }}},
        {'entity_id': 'B1', 'team': '蓝', 'type': '英雄', 'zone': '中央高地',
         'hp_ratio': .68, 'state': {'attack_status': 'possible_attack'}},
    ], zone_control={'contested_zones': ['中央高地']})
    analysis['current_mode'] = '防守'
    analysis['detailed_posture'] = {
        'code': 'contested_zone', 'label': '关键区域争夺',
        'objective': '围绕中央高地集中协同，避免拆散兵力',
        'evidence_conditions': ['敌我单位同时出现在中央高地'],
        'source': 'evidence_rule_derived_posture',
        'trigger': '出现可确认目标且支援路线可达',
        'fallback': '退回相邻己方功能区',
    }
    analysis['forecast_context'] = {'predictions': [{
        'horizon_seconds': 10,
        'global_mode': {'top': '进攻', 'confidence': .64},
        'opponent_trend': {'top': 'press', 'confidence': .71},
        'entities': [{'side': 'opponent', 'entity_id': 'B1', 'entity_type': '英雄',
                      'from_zone': '中央高地', 'to_zone': '红前哨区', 'presence': .67}],
    }]}

    result = build_global_narrative(analysis)
    sections = result['sections']
    own, enemy = sections['unit_analysis']

    assert result['description'].startswith('现况：')
    assert '我方步兵3交战（中央高地）' in result['description']
    assert '敌方英雄疑似接敌（中央高地）' in result['description']
    assert sections['current_situation']['own_side'].startswith('我方重点：')
    assert sections['current_situation']['opponent_side'].startswith('敌方重点：')
    assert sections['current_situation']['side_key_events']['friendly'][0]['entity_id'] == 'R3'
    assert sections['current_situation']['side_key_events']['friendly'][0]['event'] == 'attacking'
    assert sections['current_situation']['side_key_events']['friendly'][0]['event_label'] == '观测到正在攻击'
    assert sections['current_situation']['side_key_events']['opponent'][0]['entity_id'] == 'B1'
    assert sections['current_situation']['side_key_events']['opponent'][0]['event'] == 'possible_contact'
    assert sections['current_situation']['side_key_events']['opponent'][0]['event_label'] == '距离范围内疑似接敌，目标未确认'
    assert sections['current_situation']['tactical_stage']['label'] == '争夺关键区'
    assert '敌我单位同时出现在中央高地' in sections['current_situation']['tactical_stage']['evidence']
    assert result['forecast_analysis'].startswith('预测解析：')
    assert '+10秒' in result['forecast_analysis'] and '置信度' in result['forecast_analysis']
    assert sections['overall_recommendation']['detailed_posture'] == '关键区域争夺'
    assert sections['overall_recommendation']['layers']['strategic']['evaluation']['status'] == 'observed_response_proxy'

    assert own['unit_ref']['entity_id'] == 'R3'
    assert own['analysis_role'] == 'friendly_recommendation'
    assert own['current_status'].startswith('步兵3在中央高地')
    assert own['recommended_action'] == '留在中央高地掩护，不单独追击'
    assert '尚无传感器指定对象' in own['recommendation_reason']
    assert own['future_prediction'] is None

    assert enemy['entity_id'] == 'B1'
    assert enemy['analysis_role'] == 'enemy_forecast'
    assert enemy['current_status'].startswith('敌方英雄')
    assert '约10秒后' in enemy['future_prediction']
    assert '中央高地到红前哨区' in enemy['future_prediction']
    assert enemy['recommended_action'] is None


def test_forecast_brief_uses_confidence_for_the_specific_prediction_head():
    narrative = build_global_narrative({
        'current_mode': '均衡',
        'state': {'team': '红'},
        'forecast_context': {'predictions': [{
            'horizon_seconds': 10,
            # A high-confidence mode output must not inflate the confidence of
            # the separately uncertain enemy movement trend.
            'global_mode': {'top': '进攻', 'confidence': .92},
            'opponent_trend': {'top': '推进', 'confidence': .31},
        }]},
    })

    assert '时序模型' in narrative['forecast_analysis']
    assert '整体态势可能偏进攻' in narrative['forecast_analysis']
    assert '92%' in narrative['forecast_analysis']
    assert '敌方可能推进' not in narrative['forecast_analysis']


def test_forecast_brief_explains_each_horizon_with_its_own_confidence_and_use():
    narrative = build_global_narrative({
        'current_mode': '均衡',
        'state': {'team': '红'},
        'forecast_context': {'predictions': [
            {'horizon_seconds': 120},
            {'horizon_seconds': 30,
             'opponent_trend': {'top': 'press', 'confidence': .65},
             'enemy_region': {'top': '红前哨区', 'confidence': .73}},
            {'horizon_seconds': 10,
             'global_mode': {'top': '进攻', 'confidence': .92},
             'opponent_trend': {'top': 'press', 'confidence': .31}},
            {'horizon_seconds': 60,
             'opponent_trend': {'top': 'retreat', 'confidence': .44}},
        ]},
    })

    brief = narrative['forecast_analysis']
    assert '+10秒整体态势可能偏进攻（整体态势置信度92%，短时路线参考）' in brief
    assert '+30秒敌方可能转向红前哨区（区域置信度73%，局部控制准备）' in brief
    assert '+60秒趋势置信度低，先观察' in brief
    assert '+120秒方向不确定' in brief
    # The low-confidence trend is not inflated by the high-confidence mode.
    assert '+10秒敌方可能推进' not in brief
    assert brief.index('+10秒') < brief.index('+30秒') < brief.index('+60秒') < brief.index('+120秒')
    assert len(brief) < 180


def test_posture_category_has_evaluable_observed_proxies_and_no_causal_claim():
    narrative = build_global_narrative({
        'current_mode': '防守',
        'state': {'team': '红'},
        'detailed_posture': {
            'code': 'deep_interception', 'label': '纵深拦截',
            'objective': '在基地入口拦截入侵单位',
            'evidence_conditions': ['发现敌方进入己方防区'],
            'source': 'evidence_rule_derived_posture',
        },
    })
    evaluation = narrative['sections']['overall_recommendation']['layers']['strategic']['evaluation']

    assert evaluation['category'] == 'deep_interception'
    assert '敌方入侵单位数变化' in evaluation['observable_proxies']
    assert evaluation['window_seconds'] == [3, 10]
    assert evaluation['causal_benefit_claimed'] is False


def test_missing_global_mode_stays_unknown_instead_of_becoming_balanced():
    narrative = build_global_narrative({'state': {'team': '红'}})

    assert '均衡' not in narrative['sections']['overall_recommendation']['detailed_posture']
    assert '当前整体局势尚不明确' in narrative['sections']['current_situation']['posture_plain_language']
    assert '保持当前区域控制' not in narrative['sections']['overall_recommendation']['objective']


def test_enemy_unit_forecast_and_friendly_unit_advice_are_distinct_output_kinds():
    narrative = build_global_narrative(_minimal_analysis([
        {'entity_id': 'r1', 'team': '红', 'type': '工程', 'zone': '红补给区',
         'state': {'decision_guidance': {
             'applicable': True, 'current_state': '携带资源回兑换',
             'action': '返回补给区兑换',
             'action_reason': '当前确认携带资源',
             'action_label': 'return_resource_to_supply',
             'trigger_condition': '持有资源且未兑换',
             'cancel_condition': '兑换完成',
         }}},
        {'entity_id': 'b1', 'team': '蓝', 'type': '英雄', 'zone': '蓝中路',
         'state': {'attack_status': 'unknown'}},
    ]))
    own = next(item for item in narrative['sections']['unit_analysis']
               if item['relation'] == '己方')
    enemy = next(item for item in narrative['sections']['unit_analysis']
                 if item['relation'] == '敌方')

    assert own['output_type'] == 'recommendation'
    assert own['recommended_action'] == '返回补给区兑换'
    assert own['future_prediction'] is None
    assert enemy['output_type'] == 'prediction'
    assert enemy['recommended_action'] is None
    assert enemy['current_status'] == enemy['current_state']
