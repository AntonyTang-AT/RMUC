import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.llm_row_scorecard import score_rows


def _issue(decision=None):
    return {
        'game_id': 1, 'issued_at_s': 10.0, 'perspective': '红',
        'visibility_policy': 'rmuc_2026_referee_radar_protocol_v1',
        'issue_state': {'t': 10.0, 'robots': {
            '7': {'entity_id': 7, 'type': '英雄', 'team': '红',
                  'position': [8.0, 7.0], 'functional_zone': '红中路',
                  'hp_ratio': 0.8},
            '2': {'entity_id': 2, 'type': '哨兵', 'team': '蓝',
                  'position': [21.0, 7.0], 'functional_zone': '蓝中路',
                  'hp_ratio': 0.9},
        }},
        'decision': decision or {'units': {'英雄': {'move': '推进', 'fire': True,
                                                   'target': '敌方哨兵'}}},
    }


def _row(t, rid='7', team='红', typ='英雄', x=8.5, hp=75):
    return {'game_id': 1, '时刻秒': float(t), '阵营': team,
            'robot_id': rid, '机器人类型': typ, 'x': x, 'y': 7.0,
            '当前血量': hp, '最大血量': 100}


def _cell(report):
    return report['results'][0]


def test_exact_same_entity_join_and_missing_join_is_unknown():
    # 13s is exact t+3; 20.01s must not be rounded into exact t+10.
    report = score_rows([_issue()], [_row(13, x=13.0), _row(20.01, x=13.5)],
        event_rows=[{'game_id': 1, '阵营': '红', 'robot_id': '7',
                     '时刻秒': 12.0, '事件类型': '发弹'},
                    {'game_id': 1, '阵营': '红', 'robot_id': '7',
                     '时刻秒': 13.0, '事件类型': '发弹'}])
    horizons = _cell(report)['horizons']
    assert horizons['3']['exact_future_row_n'] == 1
    assert horizons['3']['metrics']['movement_response']['rate'] == 1.0
    assert horizons['3']['metrics']['observed_firing_event']['rate'] == 1.0
    assert horizons['10']['exact_future_row_n'] == 0
    assert horizons['10']['exact_future_row_coverage'] == 0.0
    assert horizons['10']['unknown']['exact_future_row_missing'] == 1


def test_unconfirmed_target_never_scores_target_hp():
    report = score_rows([_issue()], [_row(13), _row(20),
        _row(13, rid='2', team='蓝', typ='哨兵', x=20, hp=0),
        _row(20, rid='2', team='蓝', typ='哨兵', x=20, hp=0)])
    for horizon in _cell(report)['horizons'].values():
        assert horizon['metrics']['target_hp_delta']['scorable_n'] == 0
        assert horizon['unknown']['target_id_unknown'] == 1


def test_unknown_perspective_and_ambiguous_unit_are_not_failures():
    issue = _issue()
    issue['perspective'] = 'both_sides'
    report = score_rows([issue], [_row(13)])
    result = _cell(report)
    assert result['side'] == 'unknown'
    assert result['unknown']['decision_perspective_unknown'] == 1
    assert result['horizons']['3']['exact_future_row_n'] == 0

    issue = _issue()
    issue['issue_state']['robots']['8'] = dict(issue['issue_state']['robots']['7'])
    report = score_rows([issue], [_row(13)])
    assert _cell(report)['unknown']['unit_identity_ambiguous_or_missing'] == 1


def test_legacy_visibility_blocks_deployment_claim():
    issue = _issue()
    issue['visibility_policy'] = 'legacy_complete_timeseries_current_state_not_radar_masked'
    report = score_rows([issue], [_row(13)])
    assert report['visibility_audit']['deployable_claim_allowed'] is False
    assert report['visibility_audit']['legacy_omniscient_rows_n'] == 1
