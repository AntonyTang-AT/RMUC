import shutil
import inspect
import subprocess
import tempfile
from pathlib import Path

import pytest

from eval.build_battle_replay_visual import (HTML_PREFIX, _compact_current,
                                             _compact_browser_guidance,
                                             _compact_global_prediction,
                                             _matured_horizons,
                                             _replay_global_narrative,
                                              _guidance_display, _position_fusion_configuration,
                                              _argument_parser, _load_llm_replay_decisions,
                                              _share_replay_decisions, build_payload,
                                              _zone_payload)


def test_display_partition_covers_field_without_material_overlap():
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polygons = [Polygon(item["points"]) for item in _zone_payload()]
    union = unary_union(polygons)
    assert abs(union.area - 28.0 * 15.0) < 1e-3
    overlap = 0.0
    for index, polygon in enumerate(polygons):
        for other in polygons[index + 1:]:
            overlap += polygon.intersection(other).area
    assert overlap < 1e-3


def test_replay_contains_adjustable_step_control_and_no_dashed_marker_css():
    assert 'id="rm-step"' in HTML_PREFIX
    assert 'value="3"' in HTML_PREFIX and 'value="60"' in HTML_PREFIX
    assert 'stroke-dasharray:8' not in HTML_PREFIX


def test_replay_loads_only_valid_candidate_rows_for_selected_game(tmp_path):
    rows = [
        {'game_id': 7, 'perspective': '红', 'issued_at_s': 3.0,
         'adapter': 'out_m2/candidate/adapter', 'contract_valid': True,
         'strict_parse_valid': True, 'contract_repairs': [],
         'contract_decision': {'t': 3.0, 'global_mode': '进攻', 'units': {}}},
        {'game_id': 8, 'perspective': '蓝', 'issued_at_s': 3.0,
         'adapter': 'out_m2/other/adapter', 'contract_valid': True,
         'contract_decision': {'t': 3.0, 'global_mode': '防守', 'units': {}}},
        {'game_id': 7, 'perspective': '蓝', 'issued_at_s': 3.0,
         'adapter': 'out_m2/bad/adapter', 'contract_valid': False,
         'contract_decision': {'t': 3.0, 'global_mode': '防守', 'units': {}}},
    ]
    path = tmp_path / 'rows.jsonl'
    path.write_text('\n'.join(__import__('json').dumps(row, ensure_ascii=False) for row in rows),
                    encoding='utf-8')
    loaded = _load_llm_replay_decisions([path], 7)
    assert loaded[('红', 3.0)]['decision']['global_mode'] == '进攻'
    assert ('蓝', 3.0) not in loaded
    assert 'llmDecision=frame.llm_decisions?.[side]||null' in HTML_PREFIX


def test_feedback_matures_at_each_exact_horizon_before_next_decision():
    assert _matured_horizons(10, 19.9) == []
    assert _matured_horizons(10, 20) == [10]
    assert _matured_horizons(10, 40) == [10, 30]
    assert _matured_horizons(10, 130) == [10, 30, 60, 120]
    assert 'adaptiveFeedback' in HTML_PREFIX
    assert 'replayNarrative.adaptive_feedback' in HTML_PREFIX


def test_validation_position_fusion_replay_is_opt_in_and_discloses_candidate_metadata():
    args_default = _argument_parser().parse_args([])
    args_candidate = _argument_parser().parse_args(['--position-fusion-candidate'])
    assert args_default.position_fusion_candidate is False
    assert args_candidate.position_fusion_candidate is True
    off = _position_fusion_configuration(False)
    on = _position_fusion_configuration(True)
    assert off['enabled'] is False and off['model_weight_by_horizon'] is None
    assert on['enabled'] is True
    assert on['label'] == 'validation-driven experimental group-centroid position fusion'
    assert on['model_weight_by_horizon'] == {'10s': .25, '30s': .75, '60s': .75, '120s': 1.0}
    assert 'yaw excluded' in on['velocity_source']
    assert on['default_changed'] is False
    assert 'validation-driven experimental group-centroid position fusion' in HTML_PREFIX
    assert '不使用yaw作为车头方向' in HTML_PREFIX


def test_compact_current_keeps_robot_identity_for_individual_advice():
    entity = {'entity_id': 'red-hero-2', 'team': '红', 'entity_type': '英雄',
              'hp': 100, 'maxhp': 200, 'position': {'x': 3, 'y': 4},
              'source': 'sensor', 'observability_state': 'current'}
    compact = _compact_current(entity, [entity], '红')
    assert compact['entity_id'] == 'red-hero-2'
    assert compact['hp_ratio'] == .5


def test_replay_lifecycle_uses_explicit_evidence_and_does_not_infer_death_from_stillness():
    base = {'entity_id': '红:1', 'team': '红', 'entity_type': '步兵3',
            'position': {'x': 3, 'y': 4}}
    first = _compact_current({**base, 'hp': None}, [], '红')
    later_same_position = _compact_current({**base, 'hp': None}, [], '红')
    assert first['position'] == later_same_position['position']
    assert first['life_state'] == later_same_position['life_state'] == 'unknown'
    assert later_same_position['life_state_source'] == 'hp_missing'

    confirmed_dead = _compact_current({**base, 'hp': 0}, [], '红')
    assert confirmed_dead['life_state'] == 'destroyed'
    assert confirmed_dead['life_state_source'] == 'explicit_current_hp'

    # A positive current HP reading marks a respawn/active slot even when an
    # older upstream lifecycle label still says destroyed.
    respawned = _compact_current(
        {**base, 'hp': 25, 'life_state': 'destroyed'}, [], '红')
    assert respawned['life_state'] == 'active'
    assert respawned['life_state_source'] == 'explicit_current_hp'

    upstream_dead = _compact_current(
        {**base, 'hp': None, 'life_state': 'destroyed'}, [], '红')
    assert upstream_dead['life_state'] == 'destroyed'
    assert upstream_dead['life_state_source'] == 'upstream_lifecycle'


def test_guidance_display_names_action_conditions_provenance_and_calibration_limit():
    display = _guidance_display({
        'trigger_condition': 'HP比例低于0.30',
        'cancel_condition': 'HP比例恢复到0.50以上',
        'decision_scorecard': {
            'feasibility': {'status': 'scored', 'score': 100.0},
            'execution': {'status': 'unknown'},
            'consequence': {'status': 'pending', 'metrics': ['own_survival', 'own_hp_change']},
            'causal_benefit': {'status': 'not_identifiable'},
        },
        'evidence': [
            {'source': 'current_observation'},
            {'source': 'functional_zone_graph'},
            {'source': 'visible_geometry_inference'},
        ],
    })
    assert display == {
        'status': '规则建议（未校准，不代表已执行）',
        'trigger': 'HP比例低于0.30',
        'stop': 'HP比例恢复到0.50以上',
        'evidence': '传感器观测、区域图计算、几何推断',
        'feasibility': '路线可行性100%',
        'execution': '执行未知（无命令日志）',
        'consequence': '待3/10秒核对我方存活、我方血量变化',
        'causal_benefit': '收益因果未知',
    }
    assert all(label in HTML_PREFIX for label in ('满足条件', '停止条件', '可行性：',
                                                   '执行：', '后果：', '收益因果未知'))


def test_browser_guidance_compaction_keeps_every_rendered_advice_and_score_field():
    entity = {'special_state': {
        'attack_status': 'attacking',
        'evidence': 'keep current state evidence',
        'guidance_display': {'status': '规则建议', 'trigger': '触发',
                             'stop': '停止', 'evidence': '传感器观测',
                             'feasibility': '路线未知', 'execution': '待核验',
                             'consequence': '待3/10秒核对我方存活',
                             'causal_benefit': '收益因果未知'},
        'decision_guidance': {
            'applicable': True, 'action': '撤回', 'action_label': 'low_hp_retreat',
            'display_summary': {'action': '撤回', 'reason': '血量偏低',
                                'reassess_when': '到安全区'},
            'trigger_condition': '血量低于30%', 'cancel_condition': '血量恢复',
            'action_reason': '血量偏低', 'first_step_zone': '红中路',
            'movement_goal_zone': '红基地', 'target_zone': '红基地',
            'evaluation': {'scorable': True, 'metric': 'next_zone',
                           'expected': '红中路', 'target_linkage': 'explicit_sensor_target',
                           'expected_target_id': '蓝:3', 'debug': 'omit'},
            'target_unit_id': '蓝:3', 'target_unit_type': '步兵3',
            'decision_scorecard': {
                'feasibility': {'status': 'scored', 'score': 100, 'debug': 'omit'},
                'execution': {'status': 'unknown', 'debug': 'omit'},
                'consequence': {'status': 'pending', 'metrics': ['own_survival'],
                                'debug': 'omit'},
                'causal_benefit': {'status': 'not_identifiable', 'debug': 'omit'},
                'target_confirmation': {'status': 'unknown', 'debug': 'omit'},
                'debug_section': {'detail': 'omit'},
            },
            'evidence': [{'field': 'hp_ratio', 'value': .2}],
            'future_prediction': 'duplicate prose',
        },
    }}
    compacted = _compact_browser_guidance(entity)
    guidance = compacted['special_state']['decision_guidance']
    assert guidance['display_summary']['action'] == '撤回'
    assert guidance['trigger_condition'] == '血量低于30%'
    assert guidance['evaluation'] == {
        'scorable': True, 'metric': 'next_zone', 'expected': '红中路',
        'target_linkage': 'explicit_sensor_target', 'expected_target_id': '蓝:3'}
    assert guidance['movement_goal_zone'] == '红基地'
    assert guidance['target_unit_type'] == '步兵3'
    assert guidance['decision_scorecard'] == {
        'feasibility': {'status': 'scored', 'score': 100},
        'execution': {'status': 'unknown'},
        'consequence': {'status': 'pending', 'metrics': ['own_survival']},
        'causal_benefit': {'status': 'not_identifiable'},
        'target_confirmation': {'status': 'unknown'},
    }
    assert 'evidence' not in guidance and 'future_prediction' not in guidance
    assert compacted['special_state']['guidance_display']['evidence'] == '传感器观测'
    assert compacted['special_state']['attack_status'] == 'attacking'


def test_replay_global_prediction_discloses_promoted_30s_head_and_neural_comparison():
    item = _compact_global_prediction({
        'global_mode': {'top': '进攻', 'confidence': .61, 'probabilities': {'进攻': .61}},
        'global_mode_source': 'mode_change_xgb_v1',
        'global_mode_neural_baseline': {'top': '均衡', 'confidence': .59},
    })
    assert item['mode'] == '进攻'
    assert item['mode_source'] == 'mode_change_xgb_v1'
    assert item['mode_neural_baseline'] == '均衡'
    assert '原时序模型' in HTML_PREFIX and 'XGBoost' in HTML_PREFIX


def test_replay_narrative_uses_own_observed_zones_and_separate_confidences():
    current = [
        {'entity_id': 'r1', 'team': '红', 'role': 'self', 'type': '步兵3',
         'zone': '场地开放区', 'position': [10, 7], 'hp_ratio': .8,
         'life_state': 'active', 'observability': 'current',
         'special_state': {'attack_status': 'attacking',
                           'attacking_target': {'entity_id': 'b1', 'entity_type': '英雄'}}},
        {'entity_id': 'b1', 'team': '蓝', 'role': 'opponent', 'type': '英雄',
         'zone': '场地开放区', 'position': [11, 7], 'hp_ratio': .7,
         'life_state': 'active', 'observability': 'current',
         'special_state': {'attack_status': 'possible_attack',
                           'possible_target': {'entity_id': 'r1', 'distance_m': 1.0},
                           'evidence': 'visible_geometry_inference'}},
        {'entity_id': 'r-outpost', 'team': '红', 'role': 'self', 'type': '前哨站',
         'zone': '场地未知区', 'position': None, 'hp_ratio': .9,
         'life_state': 'active', 'observability': 'current', 'special_state': {}},
    ]
    forecast = {
        'mode': '均衡', 'mode_confidence': .98,
        'mode_probabilities': {'进攻': .01, '均衡': .98, '防守': .01},
        'trend': '稳定', 'trend_confidence': .521,
        'trend_probabilities': {'推进': .22, '稳定': .521, '回撤': .259},
        'region': 'x2y1', 'region_confidence': .565,
        'region_probabilities': {'x2y1': .565},
        'outlook': {'contested_zones': ['中央高地'], 'recommended_actions': [{
            'objective': '保持当前区域控制并等待可验证的意图变化',
            'trigger': '出现连续帧推进或建筑受击',
            'fallback': '回到最近己方控制区',
        }]},
    }

    result = _replay_global_narrative('红', current, forecast, 30)

    assert '前哨站在场地未知区' not in result['description']
    assert '争夺场地开放区' in result['description']
    assert '场地开放区' in result['description']
    assert result['prediction_analysis'] == (
        '+30秒（局部控制准备）：敌方趋势偏向暂时维持位置（52%）；'
        '整体态势暂时均衡（98%）；区域倾向x2y1（56%）；后续按实况核验。')
    assert result['details']['forecast_analysis']['model_source'] == '时序模型'
    assert len(result['recommendation']) <= 48
    assert result['recommendation'].startswith('建议：')
    assert result['details']['overall_recommendation']['fallback'] == '回到最近己方控制区'
    assert '场地开放区' in result['recommendation']
    assert result['details']['forecast_analysis']['horizons'][0]['mode_confidence'] == .98
    assert result['details']['forecast_analysis']['horizons'][0]['trend_confidence'] == .521
    assert 'unit_analysis' not in result['details']
    assert {'current_situation', 'forecast_analysis', 'overall_recommendation'} <= set(result['details'])
    assert result['details']['forecast_analysis']['horizons'][0]['horizon_seconds'] == 30


def test_global_narrative_discloses_online_router_state_without_claiming_training():
    result = _replay_global_narrative('红', [], {
        'mode': '均衡', 'mode_confidence': .6,
        'adaptive_by_task': {
            'global_mode': {'strategy': 'persistence', 'samples': 23,
                            'reason': 'model_underperforms_persistence'},
            'opponent_trend': {'strategy': 'model', 'samples': 20,
                               'reason': 'model_kept'},
            'enemy_region': {'strategy': 'model', 'samples': 18,
                             'reason': 'warmup'},
        },
    }, 10, include_details=False)
    assert '全局态势用持久性基线，已核验23次' in result['adaptive_feedback']
    assert '敌方趋势用时序模型，已核验20次' in result['adaptive_feedback']
    assert '不更新模型权重' in result['adaptive_feedback']


def test_compact_global_prediction_keeps_per_horizon_router_status():
    compact = _compact_global_prediction(
        {'global_mode': {'top': '防守', 'probabilities': {'防守': 1.0}}},
        {'global_mode@10s': {'strategy': 'persistence', 'samples': 12}}, 10)
    assert compact['adaptive_by_task']['global_mode']['strategy'] == 'persistence'
    assert compact['adaptive_by_task']['global_mode']['samples'] == 12


def test_replay_ui_renders_generated_narrative_and_marks_possible_attack_as_uncertain():
    assert callable(_replay_global_narrative)
    assert 'replayNarrative.description' in HTML_PREFIX
    assert 'replayNarrative.prediction_analysis' in HTML_PREFIX
    assert 'replayNarrative.recommendation' in HTML_PREFIX
    assert "targetLink==='explicit_sensor_target'?'传感器指定'" in HTML_PREFIX
    assert "targetLink==='visible_candidate_only'?'可见候选':'来源未知'" in HTML_PREFIX
    assert "?targetType?`${targetSource}：${targetType}" in HTML_PREFIX
    assert ":'攻击对象：未确认':'';" in HTML_PREFIX
    assert '<details class="narrative-evidence">' in HTML_PREFIX
    assert '附近有疑似目标（距离推断）' in HTML_PREFIX
    assert "MOBILE_TYPES.includes(e.type)" in HTML_PREFIX
    assert 'const unitRows=visibleUnits.map(unit=>' in HTML_PREFIX
    assert '当前状态</th>' in HTML_PREFIX and '敌方未来预测 / 我方建议动作与理由' in HTML_PREFIX


def test_replay_draw_initializes_runtime_types_and_reuses_static_render_inputs():
    # These values are immutable for a replay session; define them once before
    # the first draw so the global narrative and unit table can render.
    assert 'const MOBILE_TYPES=DATA.mobile_types||[];' in HTML_PREFIX
    assert 'const zonesById=new Map();' in HTML_PREFIX
    assert 'narrativeEvidenceCache=new WeakMap()' in HTML_PREFIX
    assert 'zonesById.get(canonicalTarget)' in HTML_PREFIX
    assert 'narrativeEvidenceCache.get(narrative)' in HTML_PREFIX


def test_unit_rows_use_action_reason_reassessment_and_keep_dead_branch():
    assert 'const compact=g.display_summary||{}' in HTML_PREFIX
    assert "infoLine('建议动作',actionText)" in HTML_PREFIX
    assert "infoLine('理由与触发条件',rationale)" in HTML_PREFIX
    assert "infoLine('停止/重算'" in HTML_PREFIX
    assert "infoLine('未来预测',futureText)" in HTML_PREFIX
    assert "infoLine=(label,value)=>`<div class=\"unit-analysis-line\"" in HTML_PREFIX
    assert '先到${step&&step!==unit.zone?' in HTML_PREFIX
    assert "targetLink==='explicit_sensor_target'?'传感器指定'" in HTML_PREFIX
    assert "targetLink==='visible_candidate_only'?'可见候选':'来源未知'" in HTML_PREFIX
    assert '未来${h}秒候选路线：${route}；只是可能去向，到时核对' in HTML_PREFIX
    assert '<details class="unit-guidance-meta"><summary>评分与证据</summary>' in HTML_PREFIX
    assert "if(dead)analysis=infoLine('状态','已阵亡；停止未来预测和行动建议')" in HTML_PREFIX
    assert "role==='self'&&!dead&&guidanceDisplay" in HTML_PREFIX


def test_actual_replay_builder_attaches_narrative_to_each_horizon():
    source = inspect.getsource(build_payload)
    assert '_replay_global_narrative(' in source
    assert '"narrative": narrative' in source
    assert '"narratives": frame_narratives' in source


def test_replay_script_parses_and_gates_future_scores():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js unavailable for replay JavaScript syntax check')
    script = HTML_PREFIX.split('<script>', 1)[1].split('</script>', 1)[0]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'replay.js'
        path.write_text(script, encoding='utf-8')
        subprocess.run([node, '--check', str(path)], check=True, capture_output=True)
    assert 'frame.t>=Number(view.forecast_issued_at)+Number(h)' in script
    assert 'scoreText=matured?' in script
    assert '预测解析' in script
    assert '整体建议' in script
    assert '战术判断' not in script
    assert script.count('<h4>') == 4
    assert '微调模型决策' in script
    assert '当前观察：' not in script
    assert 'replayNarrative.prediction_analysis' in script
    assert '展开现状依据、各预测概率和建议条件' in script
    assert 'replayNarrative.recommendation' in script
    assert '满足条件' in script
    assert "infoLine('建议动作',actionText)" in script
    assert "own_firing_event:'本机发弹记录（仅行为响应）'" in script
    assert "infoLine('理由与触发条件',rationale)" in script
    assert "infoLine('停止/重算'" in script
    assert "infoLine('未来预测',futureText)" in script
    assert '先到${step&&step!==unit.zone?' in script
    assert "targetLink==='explicit_sensor_target'?'传感器指定'" in script
    assert "targetLink==='visible_candidate_only'?'可见候选':'来源未知'" in script
    assert "g.action_label==='no_action_evidence'" in script


def test_frames_share_immutable_decisions_but_keep_per_second_observations():
    decision = {
        'forecast_issued_at': 0.0,
        'adaptive_strategy': {'strategy': 'hold'},
        'horizons': {'10': {'forecast': {'mode': '推进'}, 'entities': {
            'self:工程': {'position': {'x': 2.0, 'y': 3.0},
                          'current_position': [1.0, 1.0]},
        }}},
    }
    views = {'红': decision, '蓝': decision}
    frames = [
        {'t': 0.0, 'decision_time': 0.0, 'decision_updated': True,
         'views': {'红': {'side': '红', 'visible_entities': [{'position': [1, 1]}],
                          'visibility': {'visible': 1}, 'horizons': decision['horizons'],
                          'narratives': {'10': {'description': '现况'}} ,
                          'forecast_issued_at': 0.0, 'adaptive_strategy': {}},
                   '蓝': {'side': '蓝', 'visible_entities': [], 'visibility': {},
                          'horizons': decision['horizons'], 'forecast_issued_at': 0.0,
                          'narratives': {'10': {'description': '现况'}} ,
                          'adaptive_strategy': {}}}},
        {'t': 1.0, 'decision_time': 0.0, 'decision_updated': False,
         'views': {'红': {'side': '红', 'visible_entities': [{'position': [1.5, 1]}],
                          'visibility': {'visible': 1}, 'horizons': decision['horizons'],
                          'narratives': {'10': {'description': '第二帧现况'}},
                          'forecast_issued_at': 0.0, 'adaptive_strategy': {}},
                   '蓝': {'side': '蓝', 'visible_entities': [], 'visibility': {},
                          'horizons': decision['horizons'], 'forecast_issued_at': 0.0,
                          'adaptive_strategy': {}}}},
    ]
    payload = {'frames': frames}
    compact = _share_replay_decisions(payload, {'红': {0.0: decision}, '蓝': {0.0: decision}})
    assert compact['frames'][0]['decision_ref'] == compact['frames'][1]['decision_ref'] == 0
    assert compact['frames'][0]['views']['红']['visible_entities'] != \
        compact['frames'][1]['views']['红']['visible_entities']
    assert compact['frames'][0]['views']['红']['narratives']['10']['description'] == '现况'
    assert compact['frames'][1]['views']['红']['narratives']['10']['description'] == '第二帧现况'
    assert 'horizons' not in compact['frames'][1]['views']['红']
    assert compact['decisions'][0]['views']['红']['horizons']['10']['forecast']['mode'] == '推进'
    assert 'DATA.decisions?.[frame.decision_ref]' in HTML_PREFIX
    assert 'currentAnchor(view,slot)' in HTML_PREFIX
    assert "currentSlot.length&&currentSlot.every(e=>e.life_state==='destroyed')" in HTML_PREFIX
    assert "e.life_state==='destroyed'?'team-destroyed'" in HTML_PREFIX
    assert 'if(p.forecast_suppressed||!p.position||p.presence<.25)continue' in HTML_PREFIX
