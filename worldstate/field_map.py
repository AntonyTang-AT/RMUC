# -*- coding: utf-8 -*-
"""RMUC 2026 场地通行性策略接口。

这个模块只负责读取规则对齐的通行性策略，不伪造 CAD 碰撞几何：
- ``passability_for_class`` 给出类别/机器人组合的允许、条件或禁止；
- ``boundary_class`` 只判断场外和保守边界带；
- 具体点位必须叠加 CAD/现场标定后的 geometry overlay，未覆盖点默认 unknown。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DEFAULT_SPEC = Path(__file__).resolve().parents[1] / 'docs' / 'field_map' / 'passability_map.json'

_ROBOT_ALIASES = {
    '英雄': 'hero', 'hero': 'hero',
    '步兵': 'infantry', '步兵3': 'infantry', '步兵4': 'infantry', 'infantry': 'infantry',
    '工程': 'engineer', '工程机器人': 'engineer', 'engineer': 'engineer',
    '哨兵': 'sentinel', 'sentinel': 'sentinel',
    '空中': 'air', '空中机器人': 'air', 'air': 'air',
}


def load_passability_spec(path: str | Path | None = None) -> dict[str, Any]:
    """Load the checked-in rule-aligned policy document."""
    p = Path(path) if path is not None else _DEFAULT_SPEC
    with p.open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def normalize_robot_type(robot_type: str) -> str:
    """Map replay/Chinese unit names to the policy key."""
    return _ROBOT_ALIASES.get(str(robot_type), str(robot_type))


def passability_for_class(class_name: str, robot_type: str) -> dict[str, str]:
    """Return ``allowed``/``conditional``/``forbidden`` for a zone class.

    Unknown classes and robots are deliberately forbidden until a geometry and
    robot envelope are supplied. This prevents a planner from routing through
    an unverified structure just because a point falls inside the field box.
    """
    spec = load_passability_spec()
    robot = normalize_robot_type(robot_type)
    policy = spec.get('robot_policy', {}).get(robot)
    if not policy:
        return {'status': 'forbidden', 'reason': f'unknown_robot_type:{robot_type}'}
    if class_name in policy.get('allowed', []):
        return {'status': 'allowed', 'reason': 'rule_aligned_default'}
    if class_name in policy.get('conditional', []):
        return {'status': 'conditional', 'reason': 'requires_robot_envelope_or_rule_check'}
    return {'status': 'forbidden', 'reason': 'class_not_allowed_for_robot'}


def boundary_class(x: float, y: float, margin_m: float = 0.35) -> str:
    """Conservative boundary check in the 28m x 15m planning frame.

    Interior points are intentionally returned as ``unknown`` because the
    module map is not a collision mesh. A CAD/field geometry overlay should
    replace this result before production route planning.
    """
    spec = load_passability_spec()
    frame = spec['coordinate_frame']
    w, h = float(frame['width_m']), float(frame['height_m'])
    x, y, margin_m = float(x), float(y), max(0.0, float(margin_m))
    if x < 0.0 or x > w or y < 0.0 or y > h:
        return 'outside_or_wall'
    if x <= margin_m or x >= w - margin_m or y <= margin_m or y >= h - margin_m:
        return 'outside_or_wall'
    return 'unknown'


def point_requires_geometry_overlay(x: float, y: float) -> bool:
    """Whether a point still needs CAD/现场 geometry before routing."""
    return boundary_class(x, y) == 'unknown'
