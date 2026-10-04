from worldstate.functional_zone import (
    FUNCTIONAL_ZONE_LABELS,
    assign_functional_zone,
    candidate_moves,
    functional_zone_neighbors,
    can_move_between,
    describe_transition,
)


def test_functional_zone_labels_cover_plan_areas():
    assert '中央高地' in FUNCTIONAL_ZONE_LABELS
    assert '红基地' in FUNCTIONAL_ZONE_LABELS
    assert '蓝基地' in FUNCTIONAL_ZONE_LABELS
    assert '红上公路' in FUNCTIONAL_ZONE_LABELS
    assert '蓝下公路' in FUNCTIONAL_ZONE_LABELS


def test_metric_coordinates_map_to_functional_zones():
    assert assign_functional_zone(1.2, 7.5) == '红基地'
    assert assign_functional_zone(26.8, 7.5) == '蓝基地'
    assert assign_functional_zone(14.0, 7.5) == '中央高地'
    assert assign_functional_zone(10.0, 13.5) == '红能量机关区'
    assert assign_functional_zone(17.5, 1.2) == '蓝下公路'


def test_neighbor_graph_is_explicit_and_symmetric():
    assert '中央北隧道' in functional_zone_neighbors('中央高地')
    assert '中央高地' in functional_zone_neighbors('中央北隧道')
    assert '蓝中路' not in functional_zone_neighbors('红上装配')


def test_unit_specific_movement_constraints():
    assert can_move_between('英雄', '红', '红上公路', '中央北隧道')
    assert can_move_between('步兵3', '红', '中央北隧道', '中央高地')
    assert not can_move_between('哨兵', '红', '红中路', '中央北隧道')
    assert not can_move_between('工程', '红', '红中路', '蓝中路')
    assert can_move_between('空中', '红', '红上装配', '中央高地') is False  # 非相邻仍禁止瞬移


def test_candidate_moves_and_transition_text_are_zone_to_zone():
    moves = candidate_moves('步兵3', '红', '红上公路')
    assert '中央北隧道' in moves
    text = describe_transition('步兵3', '红上公路', '中央北隧道')
    assert '红上公路 → 中央北隧道' in text



def test_official_interaction_zones_are_not_left_unknown():
    assert assign_functional_zone(1.2, 13.5) == '红停机坪'
    assert assign_functional_zone(1.2, 1.2) == '红补给区'
    assert assign_functional_zone(26.8, 1.2) == '蓝停机坪'
    assert assign_functional_zone(26.8, 13.5) == '蓝补给区'

