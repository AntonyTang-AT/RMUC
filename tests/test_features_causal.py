import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import features


def row(game_id, robot_id, t, x):
    return {
        'game_id': game_id, 'robot_id': robot_id, '时刻秒': float(t),
        '机器人类型': '英雄', '阵营': '红', '学校名': 'A',
        '当前血量': 1000.0, '最大血量': 1000.0, 'x': float(x), 'y': 0.0, 'z': 0.0,
        '枪口朝向': 0.0, '底盘功率': 0.0, '小热量': 0.0, '大热量': 0.0,
        '累计17mm发弹': 0.0, '累计42mm发弹': 0.0, '是否易伤': 0,
        '队伍总金币': 0.0, '队伍剩余金币': 0.0,
    }


def test_closest_returns_none_before_first_observation():
    rows = [{'时刻秒': 1.0, 'x': 10}, {'时刻秒': 2.0, 'x': 20}]
    assert features._closest(rows, 0.0) is None


def test_decision_grid_starts_at_first_observation_and_never_reads_future():
    rows = [row(game_id=7, robot_id=1, t=1, x=10),
            row(game_id=7, robot_id=1, t=2, x=20),
            row(game_id=7, robot_id=1, t=4, x=40)]
    dps = features.build_decision_points(rows, 7, step=3, trend=2)
    assert [dp['t'] for dp in dps] == [1.0, 4.0]
    assert dps[0]['robots'][1]['x'] == 10
    assert dps[0]['robots'][1]['observed_at'] == 1.0
    assert dps[0]['robots'][1]['age_seconds'] == 0.0
    assert dps[1]['robots'][1]['x'] == 40
    assert all(state['observed_at'] <= dp['t']
               for dp in dps for state in dp['robots'].values())


def test_no_future_state_is_inserted_for_late_entity():
    rows = [row(game_id=7, robot_id=1, t=1, x=10),
            row(game_id=7, robot_id=2, t=4, x=40)]
    dps = features.build_decision_points(rows, 7, step=3, trend=2)
    assert 2 not in dps[0]['robots']


def test_decision_grid_does_not_extend_past_last_observation():
    rows = [row(game_id=7, robot_id=1, t=1, x=10),
            row(game_id=7, robot_id=1, t=5, x=50)]
    dps = features.build_decision_points(rows, 7, step=3, trend=2)
    assert [dp['t'] for dp in dps] == [1.0, 4.0]


def test_explicit_start_after_last_observation_returns_no_points():
    rows = [row(game_id=7, robot_id=1, t=1, x=10)]
    assert features.build_decision_points(rows, 7, step=3, start_time=2) == []


def test_trend_does_not_use_observation_older_than_window():
    rows = [row(game_id=7, robot_id=1, t=1, x=10),
            row(game_id=7, robot_id=1, t=100, x=100)]
    dps = features.build_decision_points(rows, 7, step=99, trend=15)
    assert dps[-1]['robots'][1]['dx'] == 0.0
