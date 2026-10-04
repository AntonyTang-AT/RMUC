import json
from pathlib import Path

from worldstate.field_map import (
    boundary_class,
    load_passability_spec,
    passability_for_class,
)


def test_passability_spec_has_official_dimensions():
    spec = load_passability_spec()
    assert spec['coordinate_frame']['width_m'] == 28.0
    assert spec['coordinate_frame']['height_m'] == 15.0
    assert spec['coordinate_frame']['origin'].startswith('场地左下角')


def test_ground_robot_policy_is_conservative():
    assert passability_for_class('road', '步兵3')['status'] == 'allowed'
    assert passability_for_class('highland', '步兵3')['status'] == 'conditional'
    assert passability_for_class('unknown', '步兵3')['status'] == 'forbidden'
    assert passability_for_class('flight_area', '步兵3')['status'] == 'forbidden'


def test_air_policy_requires_flight_area():
    assert passability_for_class('flight_area', '空中')['status'] == 'conditional'
    assert passability_for_class('outside_or_wall', '空中')['status'] == 'forbidden'


def test_boundary_is_blocked_and_interior_needs_geometry_overlay():
    assert boundary_class(-0.1, 3.0) == 'outside_or_wall'
    assert boundary_class(0.1, 3.0) == 'outside_or_wall'
    assert boundary_class(14.0, 7.5) == 'unknown'
