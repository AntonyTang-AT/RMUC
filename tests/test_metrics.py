# tests/test_metrics.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval import metrics

def test_accuracy_perfect():
    assert metrics.accuracy(['def','cruise'], ['def','cruise']) == 1.0
    assert metrics.accuracy(['def','cruise'], ['def','hitbuilding']) == 0.5

def test_f1_macro_range():
    f = metrics.f1_macro(['def','hitbuilding','cruise'], ['def','hitbuilding','def'])
    assert 0.0 <= f <= 1.0

def test_legal_rate():
    res = [{'state':'def'}, {'state':'hitbuilding'}, {'state':'bogus'}]
    assert abs(metrics.legal_rate(res) - 2/3) < 1e-9

def test_confusion_shape():
    c = metrics.confusion(['def','cruise','def'], ['def','def','cruise'], ['def','cruise'])
    assert len(c) == 2 and all(len(v) == 2 for v in c.values())
