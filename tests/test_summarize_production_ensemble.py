from eval.summarize_production_ensemble import summarize


def _metrics(accuracy, macro_f1, brier):
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'brier': brier}


def test_summary_requires_accuracy_f1_and_brier_improvements():
    report = {
        'schema_version': 'production-ensemble-eval-v1',
        'locked_test_games': 2,
        'rows': 10,
        'runtime_policy': {'enemy_region': 'transition'},
        'global': {'enemy_region': {'10': {
            'production': _metrics(.7, .3, .4),
            'persistence': _metrics(.6, .2, .5),
        }}},
        'entity': {},
    }
    result = summarize(report)
    assert result['global']['enemy_region']['10']['double_gate'] is True
    assert result['runtime_policy']['enemy_region'] == 'transition'


def test_summary_keeps_coarse_region_confidence_separate_from_exact_region():
    report = {
        'locked_test_games': 2,
        'rows': 10,
        'global': {},
        'entity': {},
        'enemy_region_marginals': {'x_band': {'30': {
            'class_labels': ['x0', 'x1', 'x2', 'x3'],
            'production': _metrics(.8, .6, .2),
            'persistence': _metrics(.7, .5, .3),
            'paired_game_bootstrap': {'metrics': {
                'accuracy': {'ci_excludes_zero_in_favor_of_production': True},
                'macro_f1': {'ci_excludes_zero_in_favor_of_production': True},
                'brier': {'ci_excludes_zero_in_favor_of_production': True},
            }},
        }}},
    }

    result = summarize(report)

    summary = result['enemy_region_marginals']['x_band']['30']
    assert summary['class_labels'] == ['x0', 'x1', 'x2', 'x3']
    assert summary['confidence_confirmed_gate'] is True
