import pytest

from data.forecast_splits import chronological_group_split


def rows():
    return [
        {'game_id': 3, '赛区': 'A', '场次号': 3, '局号': 1, '开始时间': '2026-05-03 10:00:00', '时长秒': 0},
        {'game_id': 1, '赛区': 'A', '场次号': 1, '局号': 1, '开始时间': '2026-05-01 10:00:00', '时长秒': 0},
        {'game_id': 2, '赛区': 'A', '场次号': 1, '局号': 2, '开始时间': '2026-05-01 11:00:00', '时长秒': 0},
        {'game_id': 4, '赛区': 'A', '场次号': 4, '局号': 1, '开始时间': '2026-05-04 10:00:00', '时长秒': 0},
        {'game_id': 5, '赛区': 'A', '场次号': 5, '局号': 1, '开始时间': '2026-05-05 10:00:00', '时长秒': 0},
    ]


def test_chronological_split_keeps_full_match_groups_ordered():
    split = chronological_group_split(rows(), .5, .25)
    assert split['train'] == ['A:1', 'A:3']
    assert split['validation'] == ['A:4']
    assert split['test'] == ['A:5']
    assert split['manifest']['train']['game_ids'] == [1, 2, 3]
    assert split['manifest']['validation']['game_ids'] == [4]
    assert split['manifest']['test']['game_ids'] == [5]
    assert max(split['train_order']) < min(split['validation_order'])
    assert max(split['validation_order']) < min(split['test_order'])


def test_explicit_match_id_is_respected():
    fixture = [dict(row, match_id='m1' if row['场次号'] == 1 else f"m{row['场次号']}")
               for row in rows()]
    split = chronological_group_split(fixture, .5, .25)
    assert split['train'][0] == 'm1'


def test_duplicate_game_id_across_matches_fails():
    fixture = rows()
    fixture[1]['game_id'] = 3
    with pytest.raises(ValueError, match='game_id'):
        chronological_group_split(fixture, .5, .25)


def test_missing_group_metadata_fails():
    with pytest.raises(ValueError, match='match_id'):
        chronological_group_split([{'game_id': 1, '开始时间': '2026-05-01'}], .6, .2)


def test_reconstructed_match_id_must_identify_one_team_pair():
    fixture = rows()
    fixture[0]['赛区'] = 'A'
    fixture[0]['场次号'] = 1
    fixture[0]['红方学校'] = '甲'
    fixture[0]['蓝方学校'] = '乙'
    fixture[1]['红方学校'] = '丙'
    fixture[1]['蓝方学校'] = '丁'
    with pytest.raises(ValueError, match='match_id'):
        chronological_group_split(fixture, .5, .25)


def test_split_rejects_matches_whose_play_windows_overlap():
    fixture = [
        {'game_id': 1, 'match_id': 'm1', 'start_time': '2026-05-01 10:00:00',
         'duration_seconds': 3600},
        {'game_id': 2, 'match_id': 'm2', 'start_time': '2026-05-01 10:30:00',
         'duration_seconds': 600},
        {'game_id': 3, 'match_id': 'm3', 'start_time': '2026-05-03 10:00:00',
         'duration_seconds': 600},
        {'game_id': 4, 'match_id': 'm4', 'start_time': '2026-05-04 10:00:00',
         'duration_seconds': 600},
    ]
    with pytest.raises(ValueError, match='overlap'):
        chronological_group_split(fixture, .25, .25)


def test_manifest_end_time_includes_match_duration():
    fixture = [
        {'game_id': 1, 'match_id': 'm1', 'start_time': '2026-05-01 10:00:00',
         'duration_seconds': 3600},
        {'game_id': 2, 'match_id': 'm2', 'start_time': '2026-05-02 10:00:00',
         'duration_seconds': 600},
        {'game_id': 3, 'match_id': 'm3', 'start_time': '2026-05-03 10:00:00',
         'duration_seconds': 600},
        {'game_id': 4, 'match_id': 'm4', 'start_time': '2026-05-04 10:00:00',
         'duration_seconds': 600},
    ]
    split = chronological_group_split(fixture, .25, .25)
    assert split['manifest']['train']['end_time'] == '2026-05-01T11:00:00'


def test_missing_duration_and_end_time_fails_closed():
    fixture = [{'game_id': 1, 'match_id': 'm1', 'start_time': '2026-05-01 10:00:00'}]
    with pytest.raises(ValueError, match='duration'):
        chronological_group_split(fixture, .6, .2)
