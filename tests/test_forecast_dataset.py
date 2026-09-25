from data.forecast_dataset import build_game_forecast_examples


def row(team, robot_id, t, x, highlight=0):
    return {
        'game_id': 3, '阵营': team, 'robot_id': robot_id, '时刻秒': float(t),
        '机器人类型': '步兵3', '是否易伤': highlight,
        'x': float(x), 'y': 1.0, 'z': 0.0, '当前血量': 100.0-t,
        '最大血量': 100.0, '枪口朝向': 0.0, '底盘功率': 1.0,
        '小热量': 0.0, '大热量': 0.0, '累计17mm发弹': 0.0,
        '累计42mm发弹': None, '队伍总金币': 1.0, '队伍剩余金币': 1.0,
    }


def test_dataset_pairs_radar_masked_inputs_with_global_exact_future_labels():
    rows = []
    for t in range(14):
        rows.extend([row('红', 1, t, 1+t),
                     row('蓝', 2, t, 9-t, highlight=1 if t in (0, 6, 12) else 0)])
    examples = build_game_forecast_examples(rows, 3, horizons=(10,), step=3)
    red_t0 = examples['红'][0]
    assert red_t0['issued_at'] == 0.0
    assert '蓝:2' in red_t0['observable']['opponent_entities']
    assert red_t0['observable']['opponent_entities']['蓝:2']['position'][0] == 9.0
    assert red_t0['targets'][10]['target_at'] == 10.0
    assert red_t0['targets'][10]['truth']['entities']['蓝:2']['position'][0] == -1.0
    red_t3 = examples['红'][1]
    assert red_t3['issued_at'] == 3.0
    assert '蓝:2' in red_t3['observable']['opponent_entities']
    assert red_t3['observable']['opponent_entities']['蓝:2']['position'][0] == 6.0
    assert red_t3['observable']['opponent_entities']['蓝:2']['radar_highlighted'] is False
    assert red_t3['targets'][10]['target_at'] == 13.0
    assert examples['红'][2]['targets'][10] is None
    blue_t0 = examples['蓝'][0]
    assert blue_t0['side'] == '蓝'
    assert '红:1' in blue_t0['observable']['opponent_entities']


def test_dataset_rejects_mixed_game_rows():
    rows = [row('红', 1, 0, 1), row('蓝', 2, 0, 9)]
    rows[1]['game_id'] = 4
    try:
        build_game_forecast_examples(rows, 3)
    except ValueError as exc:
        assert 'different game_id' in str(exc)
    else:
        raise AssertionError('mixed-game input should be rejected')
