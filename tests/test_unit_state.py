from worldstate.unit_state import derive_unit_special_state


def test_combat_state_reports_attack_and_possible_target():
    entity = {'entity_id': 'r1', 'team': '红', 'entity_type': '步兵3',
              'role': 'self', 'position': [10.0, 7.0], 'firing': True}
    enemy = {'entity_id': 'b1', 'team': '蓝', 'entity_type': '英雄',
             'role': 'opponent', 'position': [11.0, 7.2]}
    state = derive_unit_special_state(entity, [entity, enemy], self_team='红')
    assert state['attack_status'] == 'attacking'
    assert state['attacking_target'] is None
    assert state['possible_target']['entity_id'] == 'b1'
    assert state['possible_target']['reason'] == 'nearest_visible_enemy'
    assert state['attacking_energy_unit'] is None


def test_engineer_resource_state_is_explicit_but_fails_closed():
    entity = {'entity_id': 'r2', 'team': '红', 'entity_type': '工程',
              'role': 'self', 'position': [1.2, 1.2], 'has_energy_unit': True}
    state = derive_unit_special_state(entity, [entity], self_team='红')
    assert state['resource_status'] == 'at_supply_area_with_energy_unit'
    assert state['has_energy_unit'] is True
    assert state['resource_target_zone'] == '红补给区'
    assert state['resource_phase'] == 'exchanging'
    assert state['resource_route']['to_zone'] == '红补给区'


def test_sentinel_and_air_special_states():
    sentinel = {'entity_id': 's', 'team': '红', 'entity_type': '哨兵',
                'role': 'self', 'position': [5.8, 7.5]}
    air = {'entity_id': 'a', 'team': '红', 'entity_type': '空中',
           'role': 'self', 'position': [4.0, 13.5], 'z': 1.2}
    s = derive_unit_special_state(sentinel, [sentinel], self_team='红')
    a = derive_unit_special_state(air, [air], self_team='红')
    assert s['defense_status'] == 'unknown'
    assert s['defense_role_prior'] == 'guarding_outpost'
    assert s['defense_evidence'] == 'functional_zone_prior'
    assert a['flight_status'] == 'airborne'
    assert a['attack_status'] == 'unknown'
    assert s['decision_guidance']['action_label'] == 'no_action_evidence'
    assert s['decision_guidance']['evaluation']['reason'] == 'location_prior_is_not_a_defense_order'
    assert any(e['field'] == 'defense_role_prior' and e['interpretation'] == 'role_prior_only'
               for e in s['decision_guidance']['evidence'])


def test_opponent_radar_does_not_turn_heat_into_direct_attack():
    entity = {'entity_id': 'b1', 'team': '蓝', 'entity_type': '英雄',
              'role': 'opponent', 'position': [10.0, 7.0], 'small_heat': 50}
    own = {'entity_id': 'r1', 'team': '红', 'entity_type': '步兵3',
           'role': 'self', 'position': [11.0, 7.0]}
    state = derive_unit_special_state(entity, [entity, own], self_team='红')
    assert state['is_attacking'] is None
    assert state['attack_status'] == 'possible_attack'
    assert state['evidence'] == 'visible_geometry_inference'
    assert 'firing' in state['unknown_fields']
    assert 'has_energy_unit' in state['unknown_fields']


def test_special_state_forecast_is_probabilistic_and_horizon_aware():
    from worldstate.unit_state import forecast_special_state
    current = {'attack_status': 'attacking', 'attacking_target': {'entity_id': 'b1'},
               'possible_target': {'entity_id': 'b1'}, 'attacking_energy_unit': False}
    near = forecast_special_state('步兵3', current, 10)
    far = forecast_special_state('步兵3', current, 120)
    assert near['attack_status_probabilities']['attacking'] > far['attack_status_probabilities']['attacking']
    assert near['target']['entity_id'] == 'b1'
    assert abs(sum(near['attack_status_probabilities'].values()) - 1.0) < 1e-6


def test_low_health_retreat_accepts_raw_rm_maxhp_and_overrides_possible_attack():
    from worldstate.functional_zone import assign_functional_zone
    from worldstate.unit_state import derive_unit_guidance

    entity = {'entity_id': 'r-low', 'team': '红', 'entity_type': '步兵3',
              'role': 'self', 'position': [10.0, 7.5], 'hp': 20, 'maxhp': 200}
    zone = assign_functional_zone(*entity['position'])
    guidance = derive_unit_guidance(
        entity,
        {'functional_zone': zone, 'attack_status': 'possible_attack'},
        [entity], self_team='红')

    assert guidance['first_step_zone'] in guidance['action']
    assert '脱离威胁' in guidance['action']
    assert '减少暴露' not in guidance['action']
    assert guidance['action_label'] == 'retreat_low_hp'
    assert '血量10%' in guidance['short_text']
    assert guidance['first_step_zone'] in guidance['short_text']
    assert '脱离威胁' in guidance['short_text']
    assert '减少暴露' not in guidance['short_text']
    assert guidance['first_step_zone'] != zone
    assert guidance['first_step_zone'].startswith('红')
    assert any(item.get('field') == 'hp_ratio' and item.get('value') == 0.1
               for item in guidance['evidence'])


def test_engineer_carrying_energy_unit_gets_known_supply_return_route():
    from worldstate.unit_state import derive_unit_guidance

    entity = {'entity_id': 'r-engineer', 'team': '红', 'entity_type': '工程',
              'role': 'self', 'position': [10.0, 7.5], 'has_energy_unit': True}
    state = derive_unit_special_state(entity, [entity], self_team='红')
    guidance = state['decision_guidance']

    assert state['resource_target_zone'] == '红补给区'
    assert guidance['target_zone'] == '红补给区'
    assert guidance['first_step_zone'] in guidance['path_zones']
    assert guidance['first_step_zone'] != guidance['path_zones'][0]
    assert guidance['action_label'] == 'return_resource_to_supply'
    assert '已确认携带资源' in guidance['short_text']
    scorecard = guidance['decision_scorecard']
    assert scorecard['feasibility']['status'] == 'scored'
    assert scorecard['feasibility']['score'] == 100.0
    assert scorecard['execution']['status'] == 'unknown'
    assert scorecard['execution']['coverage'] == 0.0
    assert scorecard['consequence']['resource_completion']['status'] == 'unknown'
    assert scorecard['causal_benefit']['status'] == 'not_identifiable'


def test_nonmovement_advice_is_feasible_without_inventing_execution_or_accuracy():
    from worldstate.functional_zone import assign_functional_zone
    from worldstate.unit_state import derive_unit_guidance

    own = {'entity_id': 'r1', 'team': '红', 'entity_type': '步兵3', 'role': 'self',
           'position': [10.0, 7.0]}
    enemy = {'entity_id': 'b1', 'team': '蓝', 'entity_type': '英雄', 'role': 'opponent',
             'position': [11.0, 7.0]}
    guidance = derive_unit_guidance(
        own, {'functional_zone': assign_functional_zone(10.0, 7.0),
              'attack_status': 'possible_attack',
              'possible_target': {'entity_id': 'b1', 'entity_type': '英雄', 'distance_m': 1.0}},
        [own, enemy], self_team='红')

    scorecard = guidance['decision_scorecard']
    assert scorecard['feasibility']['status'] == 'not_required'
    assert scorecard['feasibility']['score'] is None
    assert scorecard['execution'] == {
        'status': 'unknown', 'coverage': 0.0,
        'reason': 'controller command/acknowledgement logs are unavailable',
        'observed_behavior_proxy_is_execution': False,
    }
    assert scorecard['consequence']['target_confirmation']['status'] == 'unknown'
    assert scorecard['causal_benefit']['status'] == 'not_identifiable'


def test_unsupported_generic_unit_state_does_not_invent_a_hold_action():
    from worldstate.unit_state import derive_unit_guidance

    guidance = derive_unit_guidance(
        {'team': '红', 'entity_type': '步兵3', 'hp': 100, 'maxhp': 200},
        {'functional_zone': '红中路', 'attack_status': 'unknown'},
        entities=[], self_team='红')
    assert guidance['action_label'] == 'no_action_evidence'
    assert guidance['action'] is None
    assert guidance['applicable'] is False
    assert '暂无足够证据' in guidance['short_text']
    assert '收到明确任务信号' in guidance['trigger_condition']
    assert guidance['evaluation']['scorable'] is False


def test_possible_attack_recommendation_has_auditable_action_label_and_short_copy():
    from worldstate.unit_state import derive_unit_guidance

    own = {'entity_id': 'r1', 'team': '红', 'entity_type': '步兵3',
           'role': 'self', 'position': [10.0, 7.0], 'hp': 100, 'maxhp': 200}
    enemy = {'entity_id': 'b1', 'team': '蓝', 'entity_type': '英雄',
             'role': 'opponent', 'position': [11.0, 7.0]}
    state = derive_unit_special_state(own, [own, enemy], self_team='红')
    guidance = state['decision_guidance']

    assert guidance['action_label'] == 'cover_and_confirm_target'
    assert guidance['target_unit_id'] == 'b1'
    assert '1.0米' in guidance['short_text']
    assert guidance['first_step_zone'] == guidance['movement_goal_zone'] or guidance['movement_goal_zone'] is None
    assert guidance['first_step_distance_m'] == 0.0
    assert guidance['evaluation']['status'] == 'unknown'
    assert guidance['evaluation']['scorable'] is False
    assert '不单独追击' in guidance['action']


def test_confirmed_attack_scores_target_outcome_without_claiming_visibility_or_causality():
    from worldstate.functional_zone import assign_functional_zone
    from worldstate.unit_state import derive_unit_guidance

    own = {'entity_id': 'r1', 'team': '红', 'entity_type': '步兵3', 'role': 'self',
           'position': [10.0, 7.0]}
    enemy = {'entity_id': 'b1', 'team': '蓝', 'entity_type': '英雄', 'role': 'opponent',
             'position': [11.0, 7.0]}
    guidance = derive_unit_guidance(
        own, {'functional_zone': assign_functional_zone(10.0, 7.0),
              'attack_status': 'attacking',
              'attacking_target': {'entity_id': 'b1', 'entity_type': '英雄', 'distance_m': 1.0,
                                   'reason': 'explicit_target'}},
        [own, enemy], self_team='红')

    assert guidance['action_label'] == 'continue_attack_visible_target'
    assert guidance['target_unit_id'] == 'b1'
    assert guidance['unit_ref'] == {'entity_id': 'r1', 'team': '红', 'entity_type': '步兵3'}
    assert guidance['movement_goal_zone'] is None
    assert guidance['first_step_distance_m'] == 0.0
    assert guidance['evaluation']['metric'] == 'identified_target_hp_and_survival'
    assert guidance['evaluation']['visibility_and_command_execution'] == 'unknown'
    assert guidance['evaluation']['confounded_by_other_attackers'] is True
    score = guidance['decision_scorecard']
    assert score['consequence']['metrics'] == [
        'own_firing_event', 'identified_target_hp_change', 'target_alive']
    assert score['consequence']['target_linkage'] == 'explicit_sensor_target'
    assert score['execution']['status'] == 'unknown'
    assert score['causal_benefit']['status'] == 'not_identifiable'


def test_hero_recent_42mm_fire_labels_nearby_unit_as_candidate_not_confirmed_target():
    from worldstate.unit_state import derive_unit_special_state

    hero = {'entity_id': 'r-hero', 'team': '红', 'entity_type': '英雄',
            'role': 'self', 'position': [10.0, 7.0], 'ammo_delta': 1,
            'hp': 500, 'maxhp': 500}
    nearby = {'entity_id': 'b-hero', 'team': '蓝', 'entity_type': '英雄',
              'role': 'opponent', 'position': [11.0, 7.0]}
    guidance = derive_unit_special_state(hero, [hero, nearby], self_team='红')['decision_guidance']

    assert guidance['action_label'] == 'continue_attack_visible_target'
    assert guidance['evaluation']['target_linkage'] == 'visible_candidate_only'
    assert any(e['field'] == 'target_entity_id' and e['source'] == 'visible_geometry_inference'
               for e in guidance['evidence'])
    assert '候选目标' in guidance['action']
    assert '下一次开火前' in guidance['short_text']
    assert '我方HP' in guidance['trigger_condition']
    outcome = guidance['decision_scorecard']['consequence']
    assert outcome['metrics'] == ['own_firing_event']
    assert outcome['target_linkage'] == 'unconfirmed_candidate_only'
    assert outcome['target_hp_metrics_status'] == 'not_scored_target_identity_not_confirmed'


def test_enemy_guidance_separates_observed_state_from_probabilistic_forecast():
    from worldstate.unit_state import derive_unit_special_state

    enemy = {'entity_id': 'b1', 'team': '蓝', 'entity_type': '步兵3',
             'role': 'opponent', 'position': [11.0, 7.0], 'hp': 70, 'maxhp': 100}
    own = {'entity_id': 'r1', 'team': '红', 'entity_type': '英雄',
           'role': 'self', 'position': [10.0, 7.0]}
    guidance = derive_unit_special_state(enemy, [enemy, own], self_team='红')['decision_guidance']

    assert guidance['output_role'] == 'enemy_state_and_forecast'
    assert '敌方步兵3' in guidance['current_state_description']
    assert '血量约70%' in guidance['current_state_description']
    assert '仅是可能目标' in guidance['current_state_description']
    assert guidance['action'] is None
    assert guidance['future_prediction']['kind'] == 'probabilistic_unit_state_prior'
    assert set(guidance['future_prediction']['horizons']) == {'10', '30'}
    assert guidance['future_prediction']['movement']['status'] == 'unknown'
    assert '不是已证实动作' in guidance['future_prediction']['interpretation']


def test_friendly_advice_marks_recommendation_as_advice_and_explains_reason():
    from worldstate.unit_state import derive_unit_guidance

    own = {'entity_id': 'r-low', 'team': '红', 'entity_type': '步兵3',
           'role': 'self', 'position': [10.0, 7.5], 'hp': 20, 'maxhp': 200}
    guidance = derive_unit_guidance(own, {'functional_zone': '红中路'}, [own], self_team='红')

    assert guidance['output_role'] == 'friendly_recommendation'
    assert guidance['recommendation']['type'] == 'suggestion_not_prediction'
    assert guidance['recommendation']['action'] == guidance['action']
    assert '低于撤离触发阈值30%' in guidance['recommendation']['reason']
    assert guidance['recommendation']['trigger_condition'] == guidance['trigger_condition']
    assert guidance['recommendation']['cancel_condition'] == guidance['cancel_condition']
    assert guidance['display_summary']['action'] == guidance['action']
    assert '低血' in guidance['display_summary']['reason']
    assert guidance['display_summary']['reassess_when'] == guidance['cancel_condition']
