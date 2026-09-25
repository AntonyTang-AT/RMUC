from data.radar_visibility_replay import PROVIDER_ID, build_radar_visibility_snapshots


def row(team, robot_id, t, *, highlighted=0, x=1.0, y=2.0, hp=100.0):
    return {
        'game_id': 7, '阵营': team, 'robot_id': robot_id, '时刻秒': t,
        '机器人类型': '步兵3', '是否易伤': highlighted,
        'x': x, 'y': y, 'z': 0.0, '当前血量': hp, '最大血量': 200.0,
        '枪口朝向': 45.0, '底盘功率': 10.0, '小热量': 2.0,
        '大热量': 3.0, '累计17mm发弹': 4.0, '累计42mm发弹': None,
        '队伍总金币': 100.0, '队伍剩余金币': 50.0,
    }


def test_protocol_exposes_opponent_xy_hp_coins_but_strips_hidden_details():
    snapshots = build_radar_visibility_snapshots([
        row('红', 1, 10, highlighted=0, x=1.0),
        row('蓝', 2, 10, highlighted=1, x=8.0, hp=50.0),
        row('蓝', 3, 10, highlighted=0, x=9.0, hp=70.0),
    ], game_id=7)
    red = snapshots['红'][0]
    blue = snapshots['蓝'][0]
    red_enemy_ids = [eid for eid, ent in red['entities'].items() if ent['role'] == 'opponent']
    blue_enemy_ids = [eid for eid, ent in blue['entities'].items() if ent['role'] == 'opponent']
    assert red_enemy_ids == ['蓝:2', '蓝:3']
    assert blue_enemy_ids == ['红:1']
    red_enemy = red['entities']['蓝:2']
    assert red_enemy['position'] == [8.0, 2.0]
    assert red_enemy['source'] == 'radar'
    assert red_enemy['entity_type'] == '步兵3'
    assert red_enemy['hp'] == 50.0
    assert red_enemy['team_coins_left'] == 50.0
    assert red_enemy['radar_highlighted'] is True
    assert not {'z', 'yaw', 'power', 'small_heat', 'big_heat', 'ammo17', 'maxhp'} & set(red_enemy)
    assert blue['entities']['红:1']['radar_highlighted'] is False
    assert red['visibility_provider'] == PROVIDER_ID
    assert 'zero-zero opponent position means unsent' in red['visibility_semantics']


def test_radar_tracks_are_timestamped_without_future_fill_or_stale_carry():
    snapshots = build_radar_visibility_snapshots([
        row('红', 1, 10), row('蓝', 2, 10, highlighted=1, x=8.0),
        row('红', 1, 11), row('蓝', 2, 11, highlighted=0, x=7.0),
        row('红', 1, 12), row('蓝', 2, 12, highlighted=1, x=6.0),
    ], game_id=7)
    assert [snap['issued_at'] for snap in snapshots['红']] == [10.0, 11.0, 12.0]
    assert '蓝:2' in snapshots['红'][0]['entities']
    assert snapshots['红'][1]['entities']['蓝:2']['position'][0] == 7.0
    assert snapshots['红'][2]['entities']['蓝:2']['position'][0] == 6.0
    assert all(entity['observed_at'] <= snap['issued_at']
               for side_snaps in snapshots.values() for snap in side_snaps
               for entity in snap['entities'].values())


def test_rows_from_multiple_games_are_rejected():
    rows = [row('红', 1, 10), row('蓝', 2, 10, highlighted=1)]
    rows[1]['game_id'] = 8
    try:
        build_radar_visibility_snapshots(rows, game_id=7)
    except ValueError as exc:
        assert 'different game_id' in str(exc)
    else:
        raise AssertionError('mixed-game input should be rejected')


def test_zero_zero_is_treated_as_unsent_enemy_position():
    snapshot = build_radar_visibility_snapshots([
        row('红', 1, 10), row('蓝', 2, 10, x=0.0, y=0.0, hp=0.0),
    ], game_id=7)['红'][0]
    enemy = snapshot['entities']['蓝:2']
    assert enemy['hp'] == 0.0
    assert 'position' not in enemy


def test_opponent_drone_hp_is_omitted_because_protocol_reserves_that_slot():
    rows = [row('红', 1, 10), row('蓝', 6, 10, hp=75.0)]
    rows[1]['机器人类型'] = '空中'
    enemy = build_radar_visibility_snapshots(rows, game_id=7)['红'][0]['entities']['蓝:6']
    assert 'hp' not in enemy
    assert enemy['position'] == [1.0, 2.0]
