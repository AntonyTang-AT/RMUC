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

def test_per_class_counts_and_scores():
    y_true = ['cruise', 'def', 'cruise', 'cruise']
    y_pred = ['cruise', 'def', 'cruise', 'def']
    pc = metrics.per_class(y_true, y_pred)
    assert pc['cruise']['precision'] == 1.0
    assert abs(pc['cruise']['recall'] - 2/3) < 1e-9
    assert pc['cruise']['support'] == 3
    assert abs(pc['def']['precision'] - 0.5) < 1e-9
    assert pc['def']['recall'] == 1.0
    assert pc['def']['support'] == 1

def test_per_class_unknown_label_present_in_pred():
    y_true = ['cruise']
    y_pred = ['hitbuilding']
    pc = metrics.per_class(y_true, y_pred)
    assert 'hitbuilding' in pc
