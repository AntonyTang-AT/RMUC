# tests/test_zone.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from worldstate import zone

def test_zone_index_bounds():
    b = {'xmin': 0.0, 'xmax': 10.0, 'ymin': 0.0, 'ymax': 10.0}
    assert zone.zone_index(0, 0, b) == (0, 0)
    assert zone.zone_index(10, 10, b) == (zone.ZONE_COLS - 1, zone.ZONE_ROWS - 1)
    # 越界钳制
    assert zone.zone_index(-5, 20, b) == (0, zone.ZONE_ROWS - 1)

def test_assign_zone_returns_valid_label():
    b = {'xmin': 0.0, 'xmax': 10.0, 'ymin': 0.0, 'ymax': 10.0}
    lbl = zone.assign_zone(5.0, 5.0, b, '红')
    assert lbl in zone.ZONE_LABELS

def test_zone_labels_unique():
    assert len(set(zone.ZONE_LABELS)) == len(zone.ZONE_LABELS)
