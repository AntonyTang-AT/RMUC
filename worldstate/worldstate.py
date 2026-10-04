# worldstate/worldstate.py
from data.features import BUILDING_TYPES
from worldstate import zone as zn
from worldstate import functional_zone as fz

# ``zone`` is now the functional-zone label. Legacy 6x3 labels remain in the
# accepted vocabulary for old snapshots and downstream compatibility.
ZONE_LABELS = tuple(dict.fromkeys((*fz.FUNCTIONAL_ZONE_LABELS, *zn.ZONE_LABELS)))
accept_targets = {'敌方哨兵', '敌方英雄', '敌方步兵', '敌方前哨站', '敌方基地', '能量机关', '飞镖', '无'}


def normalize_xy(x, y, bounds):
    span_x = bounds['xmax'] - bounds['xmin']
    span_y = bounds['ymax'] - bounds['ymin']
    span_x = span_x if span_x > 1e-9 else 1e-9
    span_y = span_y if span_y > 1e-9 else 1e-9
    nx = max(0.0, min(1.0, (x - bounds['xmin']) / span_x))
    ny = max(0.0, min(1.0, (y - bounds['ymin']) / span_y))
    return (nx, ny)


def _zone_confidence(functional_zone):
    if functional_zone in ('场地未知区',):
        return 'low'
    if functional_zone in ('场地开放区',):
        return 'medium'
    return 'high'


def build_worldstate(dp, bounds):
    robots = {}
    buildings = {}
    for rid, r in dp['robots'].items():
        nx, ny = normalize_xy(r['x'], r['y'], bounds)
        ci, ri = zn.zone_index(r['x'], r['y'], bounds)
        legacy_zone = zn.zone_label(ci, ri, r['team'])
        functional_zone = fz.assign_functional_zone(r['x'], r['y'])
        entry = {
            'entity_id': rid,
            'type': r['type'], 'team': r['team'],
            'position': [round(float(r['x']), 3), round(float(r['y']), 3)],
            # Functional zone is the canonical location description. The
            # legacy grid label is retained only for old model checkpoints.
            'zone': functional_zone,
            'functional_zone': functional_zone,
            'zone_confidence': _zone_confidence(functional_zone),
            'legacy_zone': legacy_zone,
            'zone_idx': [ci, ri],
            'depth': zn.depth_of(ci, r['team']),
            'norm_xy': [round(nx, 3), round(ny, 3)],
            'hp_ratio': round(r['hp'] / r['maxhp'], 3) if r['maxhp'] else 0.0,
            'power': round(r['power'], 1), 'heat': round(r['small_heat'] + r['big_heat'], 1),
            'small_heat': round(float(r.get('small_heat') or 0.0), 1),
            'big_heat': round(float(r.get('big_heat') or 0.0), 1),
            'vuln': r['vuln'], 'ammo17': r['ammo17'], 'ammo42': r['ammo42'],
            'dfire': round(r.get('dfire', 0.0), 1),
        }
        for field in ('yaw', 'z', 'altitude', 'maxhp', 'team_coins_total',
                      'team_coins_left', 'radar_highlighted', 'firing',
                      'is_attacking', 'target_entity_id', 'target_id',
                      'has_energy_unit', 'carrying_energy_unit', 'ore_count',
                      'is_exchanging', 'exchanging'):
            if field in r:
                entry[field] = r[field]
        if r['type'] in BUILDING_TYPES:
            buildings[rid] = {'type': r['type'], 'team': r['team'], 'hp_ratio': entry['hp_ratio']}
        else:
            robots[rid] = entry
    team = {k: {'coins': v['coins_left']} for k, v in dp['team'].items()}
    return {'t': dp['t'], 'team': team, 'robots': robots, 'buildings': buildings}
