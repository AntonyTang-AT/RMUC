import pytest

from eval.forecast_report import build_forecast_manifest, summarize_readiness


MATCHES = [
    {'game_id': 1, 'match_id': 'm1', 'competition_order': 1},
    {'game_id': 2, 'match_id': 'm2', 'competition_order': 2},
]
SPLIT = {'train': ['m1'], 'validation': [], 'test': ['m2']}


def test_manifest_contains_all_games_match_groups_and_both_views():
    manifest = build_forecast_manifest(MATCHES, SPLIT, (10, 30, 60, 120),
                                       'fixture_sensor', 'label-v1')
    assert set(manifest['sides']) == {'红', '蓝'}
    assert set(manifest['all_game_ids']) == {1, 2}
    assert manifest['split_pass'] is True
    assert manifest['visibility_provider'] == 'fixture_sensor'
    assert len(manifest['manifest_sha256']) == 64


def test_manifest_rejects_missing_or_duplicate_split_groups():
    with pytest.raises(ValueError, match='assign every match'):
        build_forecast_manifest(MATCHES, {'train': ['m1'], 'test': []})
    with pytest.raises(ValueError, match='exactly once'):
        build_forecast_manifest(MATCHES, {'train': ['m1'], 'validation': ['m1'], 'test': ['m2']})


def test_manifest_allows_multiple_rounds_inside_one_match_group():
    matches = [
        {'game_id': 1, '赛区': '东部赛区', '场次号': 10, 'competition_order': 1},
        {'game_id': 2, '赛区': '东部赛区', '场次号': 10, 'competition_order': 2},
        {'game_id': 3, '赛区': '南部赛区', '场次号': 2, 'competition_order': 3},
    ]
    manifest = build_forecast_manifest(
        matches, {'train': ['东部赛区:10'], 'validation': [], 'test': ['南部赛区:2']},
    )
    assert manifest['all_match_ids'] == ['东部赛区:10', '南部赛区:2']
    assert manifest['all_game_ids'] == [1, 2, 3]


def test_missing_visibility_provider_blocks_deployment_readiness():
    manifest = build_forecast_manifest(MATCHES, SPLIT, visibility_provider=None)
    report = summarize_readiness(
        manifest,
        {'future_feature_count': 0, 'split_overlap_count': 0},
        {'invalid_count': 0, 'red_blue_complete': True},
        {'side_macro_average': {'score': .3}, 'pooled_micro': {'score': .3}},
    )
    assert report['causal_input_pass'] is True
    assert report['observable_input_pass'] is False
    assert report['split_pass'] is True
    assert report['probability_schema_pass'] is True
    assert report['red_blue_complete'] is True
    assert report['deployment_ready'] is False
    assert report['source_mode'] == 'oracle_replay'
    assert 'observable_input' in report['blocking_reasons']


def test_any_future_feature_or_bad_probability_schema_blocks_readiness():
    manifest = build_forecast_manifest(MATCHES, SPLIT, visibility_provider='radar-v1')
    report = summarize_readiness(
        manifest,
        {'future_feature_count': 1, 'split_overlap_count': 0},
        {'invalid_count': 2, 'red_blue_complete': False},
        {'side_macro_average': {}, 'pooled_micro': {}},
    )
    assert report['causal_input_pass'] is False
    assert report['probability_schema_pass'] is False
    assert report['red_blue_complete'] is False
    assert report['deployment_ready'] is False


def test_declared_visibility_mask_can_pass_observable_gate():
    manifest = build_forecast_manifest(MATCHES, SPLIT, visibility_provider='radar-mask-v1')
    report = summarize_readiness(
        manifest,
        {'future_feature_count': 0, 'split_overlap_count': 0},
        {'invalid_count': 0, 'probability_schema_pass': True, 'red_blue_complete': True},
        {'side_macro_average': {'score': .2}, 'pooled_micro': {'score': .25}},
    )
    assert manifest['source_mode'] == 'visibility_masked_replay'
    assert report['observable_input_pass'] is True
    assert report['deployment_ready'] is True
