# worldstate/zone.py
from data.features import BUILDING_TYPES

ZONE_COLS = 4
ZONE_ROWS = 3

def zone_index(x, y, bounds):
    span_x = bounds['xmax'] - bounds['xmin']
    span_y = bounds['ymax'] - bounds['ymin']
    span_x = span_x if span_x > 1e-9 else 1e-9
    span_y = span_y if span_y > 1e-9 else 1e-9
    ci = int((x - bounds['xmin']) / span_x * ZONE_COLS)
    ri = int((y - bounds['ymin']) / span_y * ZONE_ROWS)
    return (max(0, min(ci, ZONE_COLS - 1)), max(0, min(ri, ZONE_ROWS - 1)))

def zone_label(ci, ri, team):
    return f"Z{ri}{ci}"

def assign_zone(x, y, bounds, team):
    ci, ri = zone_index(x, y, bounds)
    return zone_label(ci, ri, team)

ZONE_LABELS = sorted({zone_label(ci, ri, '红') for ci in range(ZONE_COLS) for ri in range(ZONE_ROWS)})
