import pytest
import json

from eval import benchmark
from eval.benchmark import evaluate_forecast_ledger, forecast_key


def forecast(game_id, side, probabilities=None, valid_at=20):
    issued_at = 10
    return {
        'prediction_id': f'{game_id}-{side}', 'game_id': game_id, 'side': side,
        'issued_at': issued_at, 'valid_at': valid_at, 'horizon': valid_at - issued_at,
        'task': 'mode', 'entity_or_region': 'global', 'target_schema_version': 'v1',
        'input_snapshot_hash': 'abc', 'visibility_summary': {'tracks': 1},
        'model_version': 'fixture', 'calibration_version': 'none',
        'labels': ['攻', '守'], 'probabilities': probabilities or {'攻': .8, '守': .2},
    }


def test_mature_forecasts_report_both_views_side_macro_and_micro():
    predictions = [
        forecast(1, '红', {'攻': .8, '守': .2}),
        forecast(1, '蓝', {'攻': .4, '守': .6}),
    ]
    expected = [forecast_key(row) for row in predictions]
    truth = {key: '攻' for key in expected}
    report = evaluate_forecast_ledger(predictions, truth, expected, now=20, n_bootstrap=50)
    scores = report['models']['fixture']['by_task_horizon']['mode@10s']['brier']
    assert report['scored_count'] == 2
    assert report['red_blue_complete'] is True
    assert scores['red_view']['score'] == pytest.approx(.08)
    assert scores['blue_view']['score'] == pytest.approx(.72)
    assert scores['side_macro_average']['score'] == pytest.approx(.4)
    assert scores['pooled_micro']['score'] == pytest.approx(.4)
    assert scores['red_minus_blue_ci']['delta'] == '红-蓝'
    assert scores['red_minus_blue_ci']['population'] == 'complete games with both side scores'


def test_missing_predictions_are_invalid_and_never_filled_with_neutral_values():
    red = forecast(1, '红')
    expected = [forecast_key(red), forecast_key(forecast(1, '蓝'))]
    report = evaluate_forecast_ledger([red], {expected[0]: '攻', expected[1]: '攻'}, expected, now=20)
    assert report['missing_count'] == 1
    assert report['invalid_count'] == 1
    assert report['missing_rate'] == pytest.approx(.5)
    assert report['red_blue_complete'] is False
    assert report['models']['fixture']['by_task_horizon']['mode@10s']['brier']['blue_view'] is None


def test_immature_forecasts_are_deferred_until_valid_at():
    row = forecast(1, '红')
    report = evaluate_forecast_ledger([row], {forecast_key(row): '攻'}, [forecast_key(row)], now=19)
    assert report['immature_count'] == 1
    assert report['missing_count'] == 0
    assert report['scored_count'] == 0


def test_future_expected_targets_without_predictions_are_not_yet_missing():
    row = forecast(1, '红')
    report = evaluate_forecast_ledger([], {}, [forecast_key(row)], now=19,
                                      model_versions=['fixture'])
    assert report['expected_count'] == 0
    assert report['immature_expected_count'] == 1
    assert report['missing_count'] == 0


def test_censored_targets_are_excluded_from_missing_prediction_count():
    red, blue = forecast(1, '红'), forecast(1, '蓝')
    report = evaluate_forecast_ledger([], {}, [forecast_key(red), forecast_key(blue)],
                                      now=20, model_versions=['fixture'])
    assert report['scorable_expected_count'] == 0
    assert report['target_unavailable_count'] == 2
    assert report['missing_count'] == 0
    assert report['invalid_count'] == 0


def test_censoring_is_per_target_and_does_not_require_matching_entity_ids():
    red = forecast(1, '红')
    blue = forecast(1, '蓝')
    red['entity_or_region'] = 'red-hero'
    blue['entity_or_region'] = 'blue-hero'
    red_key, blue_key = forecast_key(red), forecast_key(blue)
    report = evaluate_forecast_ledger([red, blue], {red_key: '攻'}, [red_key, blue_key], now=20)
    assert report['scored_count'] == 1
    assert report['target_unavailable_count'] == 1
    assert report['invalid_count'] == 0
    assert report['red_blue_complete'] is False


def test_malformed_unhashable_target_key_is_counted_not_fatal():
    row = forecast(1, '红')
    row['entity_or_region'] = []
    report = evaluate_forecast_ledger([row], {}, [forecast_key(forecast(1, '红'))], now=20,
                                      model_versions=['fixture'])
    assert report['invalid_count'] == 1
    assert report['models']['fixture']['scored_count'] == 0


def test_integer_robot_id_is_a_valid_target_identity():
    red, blue = forecast(1, '红'), forecast(1, '蓝')
    red['entity_or_region'] = blue['entity_or_region'] = 7
    keys = [forecast_key(red), forecast_key(blue)]
    report = evaluate_forecast_ledger([red, blue], {key: '攻' for key in keys}, keys, now=20)
    assert report['scored_count'] == 2


def test_predictions_from_distinct_model_versions_are_scored_separately():
    rows = [forecast(1, side) for side in ('红', '蓝')]
    second_rows = [forecast(1, side, {'攻': .4, '守': .6}) for side in ('红', '蓝')]
    for row in second_rows:
        row['model_version'] = 'second'
    expected = [forecast_key(row) for row in rows]
    truth = {key: '攻' for key in expected}
    report = evaluate_forecast_ledger(rows + second_rows, truth, expected, now=20)
    assert report['duplicate_count'] == 0
    assert set(report['models']) == {'fixture', 'second'}
    assert report['models']['fixture']['scored_count'] == 2
    assert report['models']['second']['scored_count'] == 2


def test_malformed_and_duplicate_forecasts_are_counted_invalid():
    row = forecast(1, '红')
    blue = forecast(1, '蓝')
    key, blue_key = forecast_key(row), forecast_key(blue)
    malformed = dict(row, probabilities={'攻': .4, '守': .4})
    report = evaluate_forecast_ledger([malformed, row, row, blue],
                                      {key: '攻', blue_key: '攻'}, [key, blue_key], now=20)
    assert report['invalid_count'] == 3
    assert report['duplicate_count'] == 2
    assert report['scored_count'] == 1


def test_continuous_quantiles_use_mean_pinball_score():
    row = forecast(1, '红')
    row.update(task='position', quantiles={'0.1': 0.0, '0.9': 2.0}, bounds=[-1, 3])
    del row['labels']
    del row['probabilities']
    blue = dict(row, side='蓝')
    key, blue_key = forecast_key(row), forecast_key(blue)
    report = evaluate_forecast_ledger([row, blue], {key: 1.5, blue_key: 1.5},
                                      [key, blue_key], now=20)
    scores = report['models']['fixture']['by_task_horizon']['position@10s']['mean_pinball']
    assert scores['red_view']['score'] == pytest.approx(.1)


def test_legacy_decision_prediction_loader_requires_red_blue_index_key(tmp_path):
    path = tmp_path / 'decisions.jsonl'
    path.write_text(json.dumps({'game_id': 1, 'side': '蓝', 'index': 0}, ensure_ascii=False) + '\n',
                    encoding='utf-8')
    assert list(benchmark._load_predictions(path)) == [(1, '蓝', 0)]
    path.write_text('{"game_id": 1, "index": 0}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='side'):
        benchmark._load_predictions(path)


def test_blue_rule_view_swaps_teams_and_mirrors_field_x():
    dp = {'t': 1, 'robots': {1: {'team': '红', 'x': 2.0, 'dx': 1.0, 'yaw': 0.0}},
          'team': {'红': {'coins_left': 1}, '蓝': {'coins_left': 2}}}
    blue = benchmark._perspective_view([dp], {'xmin': 0.0, 'xmax': 10.0}, '蓝')[0]
    assert blue['robots'][1]['team'] == '蓝'
    assert blue['robots'][1]['x'] == pytest.approx(8.0)
    assert blue['robots'][1]['dx'] == pytest.approx(-1.0)
    assert blue['team']['蓝']['coins_left'] == 1
    assert dp['robots'][1]['team'] == '红'


def test_decision_benchmark_runs_and_reports_both_sides(monkeypatch, tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'split': {'eval_games': [1, 2]}}), encoding='utf-8')
    calls = []

    def fake_game(gid, db_path, mode, predictions, delta, side):
        calls.append((gid, side))
        score = .2 if side == '红' else .8
        return {'game_id': gid, 'side': side, 'n': 10, 'expected_n': 10,
                'missing_predictions': 0, 'legal_rate': 1.0,
                'conflict_rate': 0.0, 'mode_accuracy_vs_lookahead': score,
                'mode_macro_f1_vs_lookahead': score}

    monkeypatch.setattr(benchmark, '_evaluate_game', fake_game)
    report = benchmark.run_benchmark('unused', 'unused', str(manifest), ['rule'])
    model = report['models']['rule']
    assert set(calls) == {(gid, side) for gid in (1, 2) for side in ('红', '蓝')}
    assert model['red_view']['mode_accuracy_vs_lookahead'] == pytest.approx(.2)
    assert model['blue_view']['mode_accuracy_vs_lookahead'] == pytest.approx(.8)
    assert model['side_macro_average']['mode_accuracy_vs_lookahead'] == pytest.approx(.5)
    assert model['pooled_micro']['mode_accuracy_vs_lookahead'] == pytest.approx(.5)
    assert model['red_blue_complete'] is True


def test_invalid_decisions_do_not_enter_conditional_accuracy_denominator(monkeypatch):
    class FakeFeatures:
        @staticmethod
        def build_decision_points(rows, gid):
            return [{'robots': {}}, {'robots': {}}]

        @staticmethod
        def field_bounds(dps):
            return {'xmin': 0, 'xmax': 10}

    class FakeWorldstate:
        @staticmethod
        def build_worldstate(dp, bounds):
            return dp

    monkeypatch.setattr(benchmark.db, 'timeseries_for', lambda con, gid: [])
    monkeypatch.setattr(benchmark.features, 'build_decision_points', FakeFeatures.build_decision_points)
    monkeypatch.setattr(benchmark.features, 'field_bounds', FakeFeatures.field_bounds)
    monkeypatch.setattr(benchmark.worldstate, 'build_worldstate', FakeWorldstate.build_worldstate)
    monkeypatch.setattr(benchmark.strategy_label, 'strategy_series', lambda ws, delta, team: ['进攻', '防守'])
    monkeypatch.setattr(benchmark.sqlite3, 'connect', lambda path: type('Conn', (), {
        'row_factory': None, 'close': lambda self: None,
    })())
    good_decision = benchmark.sl.build_empty_decision(1.0)
    good_decision['global_mode'] = '进攻'
    bad_decision = benchmark.sl.build_empty_decision(1.0)
    bad_decision['global_mode'] = 'not-a-mode'
    good = {'decision': good_decision}
    bad = {'decision': bad_decision}
    result = benchmark._evaluate_game(1, 'unused', 'predictions',
                                      {(1, '红', 0): good, (1, '红', 1): bad}, side='红')
    assert result['legal_n'] == 2
    assert result['legal_rate'] == pytest.approx(.5)
    assert result['invalid_predictions'] == 1
    assert result['n'] == 1
    assert result['mode_accuracy_vs_lookahead'] == pytest.approx(1.0)
