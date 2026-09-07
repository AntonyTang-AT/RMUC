# tests/test_viz.py
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval import viz

def test_plot_state_strip_writes_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'strip.png')
        viz.plot_state_strip([0.0, 3.0, 6.0], ['def', 'hitbuilding', 'cruise'], p)
        assert os.path.exists(p) and os.path.getsize(p) > 0

def test_plot_heatmap_writes_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'heat.png')
        viz.plot_heatmap([0.0, 1.0, 2.0], [0.0, 0.5, 1.0], p)
        assert os.path.exists(p) and os.path.getsize(p) > 0
