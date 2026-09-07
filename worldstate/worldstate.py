# worldstate/worldstate.py
from data.features import BUILDING_TYPES
from worldstate import zone as zn

ZONE_LABELS = zn.ZONE_LABELS
accept_targets = {'敌方哨兵', '敌方英雄', '敌方步兵', '敌方前哨站', '敌方基地', '能量机关', '飞镖', '无'}

def normalize_xy(x, y, bounds):
    span_x = bounds['xmax'] - bounds['xmin']
    span_y = bounds['ymax'] - bounds['ymin']
    span_x = span_x if span_x > 1e-9 else 1e-9
    span_y = span_y if span_y > 1e-9 else 1e-9
    nx = max(0.0, min(1.0, (x - bounds['xmin']) / span_x))
    ny = max(0.0, min(1.0, (y - bounds['ymin']) / span_y))
    return (nx, ny)

def build_worldstate(dp, bounds):
    robots = {}
    buildings = {}
    for rid, r in dp['robots'].items():
        nx, ny = normalize_xy(r['x'], r['y'], bounds)
        entry = {
            'type': r['type'], 'team': r['team'],
            'zone': zn.assign_zone(r['x'], r['y'], bounds, r['team']),
            'norm_xy': [round(nx, 3), round(ny, 3)],
            'hp_ratio': round(r['hp'] / r['maxhp'], 3) if r['maxhp'] else 0.0,
            'power': round(r['power'], 1), 'heat': round(r['small_heat'] + r['big_heat'], 1),
            'vuln': r['vuln'], 'ammo17': r['ammo17'], 'ammo42': r['ammo42'],
            'dfire': round(r.get('dfire', 0.0), 1),
        }
        if r['type'] in BUILDING_TYPES:
            buildings[rid] = {'type': r['type'], 'team': r['team'], 'hp_ratio': entry['hp_ratio']}
        else:
            robots[rid] = entry
    team = {k: {'coins': v['coins_left']} for k, v in dp['team'].items()}
    return {'t': dp['t'], 'team': team, 'robots': robots, 'buildings': buildings}
