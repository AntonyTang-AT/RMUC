from data.radar_visibility_replay import build_radar_visibility_snapshots
from eval.build_battle_replay_visual import (
    HTML_PREFIX,
    _compact_current,
    _forecast_entities,
)
from runtime.temporal_forecast import suppress_destroyed_entity_forecasts


def _row(team, robot_id, kind='步兵3', *, x=1.0, y=2.0, hp=100.0):
    return {
        'game_id': 9, '阵营': team, 'robot_id': robot_id, '时刻秒': 1,
        '机器人类型': kind, '是否易伤': 0, 'x': x, 'y': y, 'z': 0.0,
        '当前血量': hp, '最大血量': 200.0, '枪口朝向': 0.0,
        '底盘功率': 10.0, '小热量': 0.0, '大热量': 0.0,
        '累计17mm发弹': 0.0, '累计42mm发弹': 0.0,
        '队伍总金币': 0.0, '队伍剩余金币': 0.0,
    }


def _forecast():
    return {
        'self:步兵3': {
            'side': 'self', 'team': '红', 'entity_type': '步兵3',
            'presence_probability': 0.9, 'position': {'x': 2.0, 'y': 2.0},
            'to_zone': '红中路', 'to_canonical_zone': 'Z01',
            'movement': {'from_zone': '红基地', 'to_zone': '红中路'},
        }
    }


def test_explicit_zero_hp_is_destroyed_but_unobservable_air_hp_is_unknown():
    rows = [
        _row('红', 1, hp=0.0),
        _row('蓝', 2, hp=0.0),
        _row('蓝', 6, '空中', hp=0.0),
        _row('蓝', 3, x=0.0, y=0.0, hp=80.0),
    ]
    red = build_radar_visibility_snapshots(rows, game_id=9)['红'][0]['entities']
    assert red['红:1']['life_state'] == 'destroyed'
    assert red['蓝:2']['life_state'] == 'destroyed'
    assert red['蓝:6']['life_state'] == 'unknown'
    assert red['蓝:3']['life_state'] == 'active'
    assert 'position' not in red['蓝:3']

    # bool is technically an int subclass in Python, but not a valid HP
    # telemetry value and must not accidentally mark a unit dead.
    bool_rows = [_row('红', 7, hp=False)]
    bool_snapshot = build_radar_visibility_snapshots(bool_rows, game_id=9)['红'][0]['entities']
    assert bool_snapshot['红:7']['life_state'] == 'unknown'
    nan_rows = [_row('红', 8, hp=float('nan'))]
    nan_snapshot = build_radar_visibility_snapshots(nan_rows, game_id=9)['红'][0]['entities']
    assert nan_snapshot['红:8']['life_state'] == 'unknown'


def test_destroyed_position_is_frozen_until_explicit_live_observation():
    rows = [
        {**_row('蓝', 2, x=1.0, y=2.0, hp=100), '时刻秒': 1},
        {**_row('蓝', 2, x=1.2, y=2.1, hp=0), '时刻秒': 2},
        {**_row('蓝', 2, x=1.6, y=2.5, hp=0), '时刻秒': 3},
        {**_row('蓝', 2, x=1.9, y=2.8, hp=None), '时刻秒': 4},
        {**_row('蓝', 2, x=5.0, y=6.0, hp=100), '时刻秒': 5},
        {**_row('蓝', 2, x=5.2, y=6.0, hp=100), '时刻秒': 6},
    ]
    snapshots = build_radar_visibility_snapshots(rows, game_id=9)
    for side in ('红', '蓝'):
        entities = [snap['entities']['蓝:2'] for snap in snapshots[side]]
        assert [e['life_state'] for e in entities[1:4]] == ['destroyed'] * 3
        assert [e['position'] for e in entities[1:4]] == [[1.2, 2.1]] * 3
        assert entities[3]['life_state_source'] == 'prior_confirmed_hp_zero'
        assert entities[2]['position_anchor_source'] == 'first_confirmed_destroyed_observation'
        for index in (1, 2, 3, 4):
            assert 'speed_mps' not in entities[index]
            assert 'motion_direction_deg' not in entities[index]
        assert entities[4]['life_state'] == 'active'
        assert entities[4]['position'] == [5.0, 6.0]
        assert entities[5]['position'] == [5.2, 6.0]
        assert entities[5]['speed_mps'] > 0
    self_track = snapshots['蓝'][3]['entities']['蓝:2']
    prediction = _forecast()
    suppress_destroyed_entity_forecasts(prediction, [self_track])
    assert prediction['self:步兵3']['forecast_suppressed'] is True


def test_missing_hp_and_hidden_air_hp_do_not_freeze_position_or_assert_death():
    rows = [
        {**_row('蓝', 6, '空中', x=1.0, hp=100), '时刻秒': 1},
        {**_row('蓝', 6, '空中', x=2.0, hp=0), '时刻秒': 2},
        {**_row('蓝', 6, '空中', x=3.0, hp=0), '时刻秒': 3},
        {**_row('红', 1, x=1.0, hp=None), '时刻秒': 1},
        {**_row('红', 1, x=2.0, hp=None), '时刻秒': 2},
        {**_row('红', 1, x=3.0, hp=None), '时刻秒': 3},
    ]
    snapshots = build_radar_visibility_snapshots(rows, game_id=9)
    enemy_air = [s['entities']['蓝:6'] for s in snapshots['红']]
    assert [e['life_state'] for e in enemy_air] == ['unknown'] * 3
    assert [e['position'][0] for e in enemy_air] == [1.0, 2.0, 3.0]
    assert all('position_anchor_source' not in e for e in enemy_air)
    own_air = [s['entities']['蓝:6'] for s in snapshots['蓝']]
    assert own_air[1]['life_state'] == own_air[2]['life_state'] == 'destroyed'
    assert own_air[1]['position'] == own_air[2]['position'] == [2.0, 2.0]
    missing_hp = [s['entities']['红:1'] for s in snapshots['红']]
    assert [e['life_state'] for e in missing_hp] == ['unknown'] * 3
    assert [e['position'][0] for e in missing_hp] == [1.0, 2.0, 3.0]


def test_only_all_explicitly_destroyed_tracks_suppress_aggregate_forecast():
    dead = {'role': 'self', 'entity_type': '步兵3', 'hp': 0}
    forecast = _forecast()
    suppress_destroyed_entity_forecasts(forecast, [dead])
    item = forecast['self:步兵3']
    assert item['forecast_suppressed'] is True
    assert item['life_state'] == 'destroyed'
    assert item['presence_probability'] == 0
    assert item['position'] is None and item['movement'] is None
    assert item['raw_predicted_zone'] is None
    assert item['zone_head_predicted_zone'] is None
    assert item['special_state_forecast'] is None

    # Some feeds carry an explicit lifecycle flag without numeric HP.
    lifecycle_only = _forecast()
    suppress_destroyed_entity_forecasts(lifecycle_only, [{
        'role': 'self', 'entity_type': '步兵3', 'hp': None,
        'life_state': 'destroyed',
    }])
    assert lifecycle_only['self:步兵3']['forecast_suppressed'] is True

    # Type-level output is still useful if another infantry is alive or a
    # stale/unknown track merely repeats its previous position.
    for active_track in (
        {'role': 'self', 'entity_type': '步兵3', 'hp': 100},
        {'role': 'self', 'entity_type': '步兵3', 'life_state': 'unknown',
         'position': [1.0, 2.0]},
    ):
        candidate = _forecast()
        suppress_destroyed_entity_forecasts(candidate, [dead, active_track])
        assert not candidate['self:步兵3'].get('forecast_suppressed', False)

    # A positive HP observation (e.g. after respawn) restores the slot.
    revived = _forecast()
    suppress_destroyed_entity_forecasts(
        revived, [{'role': 'self', 'entity_type': '步兵3', 'hp': 1}])
    assert not revived['self:步兵3'].get('forecast_suppressed', False)


def test_replay_distinguishes_unknown_stationary_from_dead_and_hides_dead_routes():
    base = {
        'entity_id': '红:1', 'team': '红', 'entity_type': '步兵3',
        'position': [1.0, 2.0], 'maxhp': 200,
    }
    stationary_unknown = _compact_current(
        {**base, 'hp': None}, [{**base, 'hp': None}], '红')
    same_position_unknown = _compact_current(
        {**base, 'hp': None}, [{**base, 'hp': None}], '红')
    assert stationary_unknown['position'] == same_position_unknown['position']
    assert stationary_unknown['life_state'] == 'unknown'
    assert _compact_current({**base, 'hp': False}, [], '红')['life_state'] == 'unknown'

    dead = _compact_current({**base, 'hp': 0}, [{**base, 'hp': 0}], '红')
    assert dead['life_state'] == 'destroyed'
    projected = _forecast()
    suppress_destroyed_entity_forecasts(projected, [
        {'role': 'self', 'entity_type': '步兵3', 'hp': 0}])
    replay = _forecast_entities({'horizons': {'10': {'entities': projected}}}, 10, [{
        'role': 'self', 'team': '红', 'type': '步兵3',
        'position': [1.0, 2.0], 'life_state': 'destroyed',
    }])['self:步兵3']
    assert replay['forecast_suppressed'] is True
    assert replay['position'] is None
    assert replay['movement'] is None

    visible = _forecast_entities({'horizons': {'10': {'entities': _forecast()}}}, 10, [{
        'role': 'self', 'team': '红', 'type': '步兵3',
        'position': [1.0, 2.0], 'life_state': 'unknown',
    }])['self:步兵3']
    assert visible['forecast_suppressed'] is False
    assert visible['position'] is not None
    assert "e.life_state==='destroyed'?'team-destroyed'" in HTML_PREFIX
    assert "if(e.life_state==='destroyed'||s.attack_status!=='attacking'||!target?.position||target.life_state==='destroyed')return ''" in HTML_PREFIX
    assert "if(entity?.life_state==='unknown'&&entity?.observability==='stale')return '信号失联'" in HTML_PREFIX
    assert "if(e.life_state==='unknown'&&e.observability==='stale')return '信号失联'" in HTML_PREFIX
    assert "e.position&&e.life_state!=='destroyed'" in HTML_PREFIX
    assert "if(currentSlot.some(e=>e.life_state==='destroyed')&&!liveAnchor)continue" in HTML_PREFIX


def test_attack_target_zero_hp_is_explicitly_dead_even_without_upstream_lifecycle(monkeypatch):
    from eval import build_battle_replay_visual as replay

    monkeypatch.setattr(replay, 'derive_unit_special_state', lambda *_args, **_kwargs: {
        'attacking_target': {'entity_id': '蓝:2', 'entity_type': '步兵3'},
    })
    attacker = {'entity_id': '红:1', 'team': '红', 'entity_type': '步兵3',
                'hp': 100, 'maxhp': 200, 'position': [1.0, 2.0]}
    dead_target = {'entity_id': '蓝:2', 'team': '蓝', 'entity_type': '步兵3',
                   'hp': 0, 'position': [3.0, 4.0]}
    compact = replay._compact_current(attacker, [attacker, dead_target], '红')
    assert compact['attack_target']['position'] == [3.0, 4.0]
    assert compact['attack_target']['life_state'] == 'destroyed'

    unknown_target = {**dead_target, 'hp': None}
    compact_unknown = replay._compact_current(attacker, [attacker, unknown_target], '红')
    assert compact_unknown['attack_target']['life_state'] == 'unknown'
