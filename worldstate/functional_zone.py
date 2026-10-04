# -*- coding: utf-8 -*-
"""基于 RMUC 2026 平面图的功能区和相邻移动图。

坐标使用场地米制坐标：x=0 红方侧，x=28 蓝方侧；y=0 下边界，y=15 上边界。
几何是规划级多边形，用于位置描述、相邻转移和预测标签；最终碰撞规划仍需 CAD/现场栅格。
"""
from __future__ import annotations

from typing import Iterable

FIELD_WIDTH_M = 28.0
FIELD_HEIGHT_M = 15.0

# Polygon vertices are ordered counter-clockwise in map coordinates. These are
# intentionally conservative functional footprints, not millimetre CAD edges.
ZONE_POLYGONS = {
    '红基地': [(0.0, 5.2), (2.8, 5.2), (2.8, 9.8), (0.0, 9.8)],
    '蓝基地': [(25.2, 5.2), (28.0, 5.2), (28.0, 9.8), (25.2, 9.8)],
    '红上装配区': [(0.0, 12.2), (7.8, 12.2), (9.5, 15.0), (0.0, 15.0)],
    '红下装配区': [(0.0, 0.0), (9.5, 0.0), (7.8, 2.8), (0.0, 2.8)],
    '蓝上装配区': [(18.5, 15.0), (28.0, 15.0), (28.0, 12.2), (20.2, 12.2)],
    '蓝下装配区': [(20.2, 2.8), (28.0, 2.8), (28.0, 0.0), (18.5, 0.0)],
    '红前哨区': [(4.8, 6.0), (7.5, 6.0), (7.5, 9.0), (4.8, 9.0)],
    '蓝前哨区': [(20.5, 6.0), (23.2, 6.0), (23.2, 9.0), (20.5, 9.0)],
    '红梯形高地区': [(4.0, 9.1), (6.8, 9.4), (7.4, 11.5), (5.0, 12.3), (3.5, 11.0)],
    '蓝梯形高地区': [(20.6, 2.7), (23.0, 2.3), (24.5, 3.5), (24.0, 5.5), (21.2, 5.2)],
    '中央北隧道': [(11.8, 11.6), (16.2, 11.6), (16.2, 12.9), (11.8, 12.9)],
    '中央南隧道': [(11.8, 2.1), (16.2, 2.1), (16.2, 3.4), (11.8, 3.4)],
    '中央高地': [(10.2, 4.0), (11.4, 2.8), (16.6, 2.8), (17.8, 4.0),
                 (17.8, 11.0), (16.6, 12.2), (11.4, 12.2), (10.2, 11.0)],
    '红上公路': [(6.8, 12.8), (11.8, 12.8), (11.8, 15.0), (8.0, 15.0)],
    '红下公路': [(8.0, 0.0), (11.8, 0.0), (11.8, 2.2), (6.8, 2.2)],
    '蓝上公路': [(16.2, 12.8), (20.0, 15.0), (16.2, 15.0)],
    '蓝下公路': [(16.2, 0.0), (20.0, 0.0), (21.2, 2.2), (16.2, 2.2)],
    '红中路': [(7.2, 3.0), (10.3, 3.8), (10.3, 11.2), (7.2, 12.0)],
    '蓝中路': [(17.7, 3.8), (20.8, 3.0), (20.8, 12.0), (17.7, 11.2)],
    # Official interaction modules from rulebook Fig. 4-4 and pages 47-58.
    # Their footprints are planning-level anchors until the STEP model is
    # converted to polygons; they take precedence over generic assembly/road.
    '红停机坪': [(0.0, 12.4), (2.8, 12.4), (2.8, 15.0), (0.0, 15.0)],
    '蓝停机坪': [(25.2, 0.0), (28.0, 0.0), (28.0, 2.6), (25.2, 2.6)],
    '红补给区': [(0.0, 0.0), (3.2, 0.0), (3.2, 2.6), (0.0, 2.6)],
    '蓝补给区': [(24.8, 12.4), (28.0, 12.4), (28.0, 15.0), (24.8, 15.0)],
    '红能量机关区': [(8.0, 12.2), (10.4, 12.2), (10.4, 14.2), (8.0, 14.2)],
    '蓝能量机关区': [(17.6, 0.8), (20.0, 0.8), (20.0, 2.8), (17.6, 2.8)],
}

FUNCTIONAL_ZONE_LABELS = tuple(ZONE_POLYGONS) + ('场地开放区', '场地未知区')

# Undirected adjacency from the floor-plan corridors and tunnel mouths.
_ADJACENCY = {
    '红基地': {'红上装配区', '红下装配区', '红前哨区'},
    '红上装配区': {'红基地', '红上公路'},
    '红下装配区': {'红基地', '红下公路'},
    '红上公路': {'红上装配区', '红中路', '红梯形高地区', '中央北隧道'},
    '红下公路': {'红下装配区', '红中路', '红梯形高地区', '中央南隧道'},
    '红前哨区': {'红基地', '红中路', '红梯形高地区'},
    '红梯形高地区': {'红上公路', '红下公路', '红中路', '红前哨区'},
    '红中路': {'红前哨区', '红梯形高地区', '红上公路', '红下公路',
                '中央北隧道', '中央南隧道'},
    '中央北隧道': {'红上公路', '红中路', '中央高地', '蓝中路', '蓝上公路'},
    '中央南隧道': {'红下公路', '红中路', '中央高地', '蓝中路', '蓝下公路'},
    '中央高地': {'中央北隧道', '中央南隧道'},
    '蓝上公路': {'蓝上装配区', '蓝中路', '蓝梯形高地区', '中央北隧道'},
    '蓝下公路': {'蓝下装配区', '蓝中路', '蓝梯形高地区', '中央南隧道'},
    '蓝前哨区': {'蓝基地', '蓝中路', '蓝梯形高地区'},
    '蓝梯形高地区': {'蓝上公路', '蓝下公路', '蓝中路', '蓝前哨区'},
    '蓝中路': {'蓝前哨区', '蓝梯形高地区', '蓝上公路', '蓝下公路',
                '中央北隧道', '中央南隧道'},
    '蓝上装配区': {'蓝基地', '蓝上公路', '蓝补给区'},
    '蓝下装配区': {'蓝基地', '蓝下公路', '蓝停机坪'},
    '蓝基地': {'蓝上装配区', '蓝下装配区', '蓝前哨区'},
    '红停机坪': {'红上装配区', '红上公路'},
    '蓝停机坪': {'蓝下装配区', '蓝下公路'},
    '红补给区': {'红下装配区', '红下公路'},
    '蓝补给区': {'蓝上装配区', '蓝上公路'},
    '红能量机关区': {'红上公路', '中央北隧道', '红上装配区'},
    '蓝能量机关区': {'蓝下公路', '中央南隧道', '蓝下装配区'},
    '场地开放区': set(),
    '场地未知区': set(),
}

# Make the hand-authored physical graph undirected even when one side was
# easier to describe from the plan.
for _source, _targets in list(_ADJACENCY.items()):
    for _target in _targets:
        _ADJACENCY.setdefault(_target, set()).add(_source)

# Canonical order used for deterministic output.
_ADJACENCY = {key: tuple(sorted(value)) for key, value in _ADJACENCY.items()}


def _point_in_polygon(x: float, y: float, polygon: Iterable[tuple[float, float]]) -> bool:
    inside = False
    vertices = list(polygon)
    if len(vertices) < 3:
        return False
    j = len(vertices) - 1
    for i, (xi, yi) in enumerate(vertices):
        xj, yj = vertices[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _clip_xy(x: float, y: float) -> tuple[float, float]:
    return (max(0.0, min(FIELD_WIDTH_M, float(x))),
            max(0.0, min(FIELD_HEIGHT_M, float(y))))


def assign_functional_zone(x: float, y: float) -> str:
    """Return the most specific functional zone for a metric map position."""
    x, y = float(x), float(y)
    if x < 0.0 or x > FIELD_WIDTH_M or y < 0.0 or y > FIELD_HEIGHT_M:
        return '场地未知区'
    # Specific modules/tunnels first, then corridors, then central highland.
    ordered = (
        '红基地', '蓝基地', '红停机坪', '蓝停机坪', '红补给区', '蓝补给区',
        '红能量机关区', '蓝能量机关区',
        '红上装配区', '红下装配区', '蓝上装配区', '蓝下装配区',
        '红前哨区', '蓝前哨区', '红梯形高地区', '蓝梯形高地区',
        '中央北隧道', '中央南隧道', '红上公路', '红下公路', '蓝上公路', '蓝下公路',
        '红中路', '蓝中路', '中央高地',
    )
    for zone_id in ordered:
        if _point_in_polygon(x, y, ZONE_POLYGONS[zone_id]):
            return zone_id
    # Interior floor not covered by a module is a valid but less specific zone.
    if 0.35 < x < FIELD_WIDTH_M - 0.35 and 0.35 < y < FIELD_HEIGHT_M - 0.35:
        return '场地开放区'
    return '场地未知区'


def assign_functional_zone_normalized(nx: float, ny: float) -> str:
    """Normalized replay coordinates ([0,1]) -> functional zone."""
    return assign_functional_zone(float(nx) * FIELD_WIDTH_M, float(ny) * FIELD_HEIGHT_M)


def functional_zone_neighbors(zone_id: str) -> tuple[str, ...]:
    """Physical neighboring zones, excluding a self-loop."""
    return _ADJACENCY.get(str(zone_id), tuple())


def _side_prefix(team: str) -> str:
    return '红' if str(team) == '红' else '蓝'


def _allowed_zone_for_unit(unit: str, team: str, zone_id: str) -> bool:
    unit = str(unit)
    team_prefix = _side_prefix(team)
    zone_id = str(zone_id)
    if zone_id in ('场地未知区', '场地开放区'):
        return unit == '空中'
    # Air robots use the same region graph for reporting, but are not blocked
    # by ground-only ownership restrictions.
    if unit in ('空中', '空中机器人'):
        return True
    # The sentinel and engineer priors are tactical/role restrictions, not walls.
    if unit in ('哨兵', 'sentinel'):
        return zone_id.startswith(team_prefix)
    if unit in ('工程', '工程机器人', 'engineer'):
        # Project prior: engineer stays on its own side; crossing the centre
        # requires a separate task-specific policy and is not a default move.
        return zone_id.startswith(team_prefix)
    return True


def can_move_between(unit: str, team: str, from_zone: str, to_zone: str) -> bool:
    """Check stay/one-edge movement plus unit-specific allowed destination."""
    from_zone, to_zone = str(from_zone), str(to_zone)
    if from_zone == to_zone:
        return _allowed_zone_for_unit(unit, team, from_zone)
    if to_zone not in functional_zone_neighbors(from_zone):
        return False
    # Enemy base/assembly footprints are interaction/restricted zones, not
    # generic transit destinations. Own-side entry is retained for return or
    # spawn actions and is still represented as a zone transition.
    own_prefix = _side_prefix(team)
    if (to_zone.endswith('基地') or to_zone.endswith('装配区')) and not to_zone.startswith(own_prefix):
        return False
    return _allowed_zone_for_unit(unit, team, to_zone)


def candidate_moves(unit: str, team: str, from_zone: str) -> tuple[str, ...]:
    """Deterministic legal next functional zones, including staying put."""
    values = [from_zone] if _allowed_zone_for_unit(unit, team, from_zone) else []
    values.extend(zone for zone in functional_zone_neighbors(from_zone)
                  if can_move_between(unit, team, from_zone, zone))
    return tuple(dict.fromkeys(values))


def describe_transition(unit: str, from_zone: str, to_zone: str) -> str:
    """Human-readable movement intent used in forecast narratives."""
    return f'{unit}：{from_zone} → {to_zone}'


def transition_record(unit: str, team: str, from_zone: str, to_zone: str,
                      probability: float | None = None) -> dict:
    """Structured movement forecast contract."""
    result = {
        'unit': str(unit), 'team': str(team),
        'from_zone': str(from_zone), 'to_zone': str(to_zone),
        'legal_adjacent': bool(can_move_between(unit, team, from_zone, to_zone)),
        'description': describe_transition(unit, from_zone, to_zone),
    }
    if probability is not None:
        result['probability'] = round(float(probability), 4)
    return result


def zone_centroid(zone_id: str) -> tuple[float, float] | None:
    """Approximate centre used to project a continuous prediction to a legal edge."""
    polygon = ZONE_POLYGONS.get(str(zone_id))
    if not polygon:
        return None
    return (sum(x for x, _ in polygon) / len(polygon),
            sum(y for _, y in polygon) / len(polygon))


def project_target_zone(unit: str, team: str, from_zone: str, raw_zone: str,
                        raw_xy: tuple[float, float] | None = None) -> tuple[str, bool]:
    """Project a continuous model zone onto the closest legal adjacent zone.

    Returns ``(target_zone, projected)``. ``projected=True`` preserves the raw
    zone in the forecast while making the actionable transition legal.
    """
    candidates = candidate_moves(unit, team, from_zone)
    if not candidates:
        return from_zone, True
    if raw_zone in candidates:
        return raw_zone, False
    if raw_xy is None:
        return candidates[0], True
    rx, ry = float(raw_xy[0]), float(raw_xy[1])
    def distance(zone_id):
        center = zone_centroid(zone_id)
        if center is None:
            return float('inf')
        return (center[0] - rx) ** 2 + (center[1] - ry) ** 2
    return min(candidates, key=distance), True
