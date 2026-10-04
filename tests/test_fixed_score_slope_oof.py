import numpy as np
import pytest

from eval.experiment_death_aware_score_slope_fixed_oof import (
    bootstrap_summary,
    fixed_class_metrics,
    labels_from_score,
    paired_delta,
)
from forecast.temporal import MODES


def test_score_thresholds_match_death_aware_mode_contract():
    got = labels_from_score(np.asarray([-1.1, -1.0, -0.1, 1.0, 1.1]))
    assert got.tolist() == [MODES.index("防守"), MODES.index("防守"),
                            MODES.index("均衡"), MODES.index("进攻"), MODES.index("进攻")]


def test_fixed_class_metrics_keep_absent_class_in_macro_scores():
    y = np.asarray([0, 0, 2, 2])
    pred = np.asarray([0, 2, 2, 2])
    result = fixed_class_metrics(y, pred)
    assert result["per_class"]["均衡"]["support"] == 0
    assert result["per_class"]["均衡"]["recall"] == 0
    assert result["balanced_accuracy"] == np.mean([0.5, 0.0, 1.0])


def test_paired_bootstrap_resamples_whole_game_metric_rows_deterministically():
    candidate = {g: {"accuracy": v, "macro_f1": v, "balanced_accuracy": v,
                     **{f"recall_{m}": v for m in MODES}} for g, v in ((1, .6), (2, .8), (3, .4))}
    reference = {g: {k: v - .1 for k, v in row.items()} for g, row in candidate.items()}
    a = paired_delta(candidate, reference, seed=7, reps=100)
    b = paired_delta(candidate, reference, seed=7, reps=100)
    assert a == b
    assert a["paired_games"] == 3
    assert a["metrics"]["accuracy"]["mean_delta"] == pytest.approx(.1)
    assert a["metrics"]["accuracy"]["ci95"] == pytest.approx([.1, .1])


def test_model_bootstrap_uses_fixed_three_class_macro_metrics():
    game_rows = {
        1: {"accuracy": .75, "macro_f1": .5, "balanced_accuracy": .5,
            "recall_进攻": 1., "recall_均衡": 0., "recall_防守": .5},
        2: {"accuracy": .25, "macro_f1": .2, "balanced_accuracy": .2,
            "recall_进攻": .2, "recall_均衡": .2, "recall_防守": .2},
    }
    result = bootstrap_summary(game_rows, seed=1, reps=100)
    assert result["accuracy"]["mean"] == .5
    assert result["recall_均衡"]["mean"] == .1
