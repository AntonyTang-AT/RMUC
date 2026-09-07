import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from game import game_theory as gt
from schema import layered as sl

def test_payoff_shape():
    assert len(gt.PAYOFF_3) == 3 and all(len(row) == 3 for row in gt.PAYOFF_3)

def test_regret_matching_valid_distribution():
    s = gt.regret_matching(gt.PAYOFF_3, iterations=800)
    assert len(s) == 3
    assert all(p >= 0 for p in s)
    assert abs(sum(s) - 1.0) < 1e-6

def test_choose_mode_returns_mode():
    m = gt.choose_mode(gt.PAYOFF_3, [1/3, 1/3, 1/3])
    assert m in sl.GLOBAL_MODES

def test_best_response_dominant():
    # 行 0 严格占优 (期望收益始终最高)
    M = [[10, 10, 10], [1, 1, 1], [0, 0, 0]]
    assert gt.best_response(M, [1/3, 1/3, 1/3]) == 0

def test_regret_matching_recovers_dominant_strategy():
    M = [[10, 10, 10], [1, 1, 1], [0, 0, 0]]
    s = gt.regret_matching(M, iterations=800)
    assert s.index(max(s)) == 0
