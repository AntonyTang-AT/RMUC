from eval.build_llm_reliability_holdout import build, evenly_spaced


def test_evenly_spaced_includes_temporal_endpoints():
    assert evenly_spaced(list(range(10)), 3) == [0, 4, 9]


def test_reliability_holdout_excludes_training_and_smoke_games():
    result = build({'game_ids': [1, 2]}, [3], list(range(1, 20)), 4)
    assert not result['training_overlap']
    assert not result['smoke_overlap']
    assert set(result['game_ids']).isdisjoint({1, 2, 3})


def test_reliability_holdout_excludes_prior_holdout_manifests():
    result = build(
        {'game_ids': [1, 2]}, [3], list(range(1, 20)), 4,
        excluded_manifests={'prior.json': {'game_ids': [4, 5, 6]}},
    )
    assert not result['prior_holdout_overlap']
    assert result['prior_holdouts']['prior.json']['game_count'] == 3
    assert set(result['game_ids']).isdisjoint({1, 2, 3, 4, 5, 6})
