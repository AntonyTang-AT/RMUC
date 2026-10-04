# tests/test_zone.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from worldstate import zone

B = {'xmin': 0.0, 'xmax': 10.0, 'ymin': 0.0, 'ymax': 10.0}


def test_zone_index_bounds():
    assert zone.zone_index(0, 0, B) == (0, 0)
    assert zone.zone_index(10, 10, B) == (zone.ZONE_COLS - 1, zone.ZONE_ROWS - 1)
    # 越界钳制
    assert zone.zone_index(-5, 20, B) == (0, zone.ZONE_ROWS - 1)


def test_assign_zone_returns_valid_label():
    lbl = zone.assign_zone(5.0, 5.0, B, '红')
    assert lbl in zone.ZONE_LABELS
    assert lbl in zone.ZONE_LABELS  # 红/蓝都落进同一 18 个语义集合


def test_zone_labels_unique():
    assert len(set(zone.ZONE_LABELS)) == len(zone.ZONE_LABELS)
    assert len(zone.ZONE_LABELS) == zone.ZONE_COLS * zone.ZONE_ROWS  # 18


def test_zone_grid_is_functional_and_rich():
    assert zone.ZONE_COLS >= 6 and zone.ZONE_ROWS >= 3
    assert zone.ZONE_COLS * zone.ZONE_ROWS >= 12  # 不多于旧 4x3


def test_depth_team_symmetry():
    # 红方主场=低x: ci=0 -> 自家深; 蓝方同样低x -> 敌家(镜像)
    assert zone.depth_of(0, '红') == '自家深'
    assert zone.depth_of(0, '蓝') == '敌家'
    assert zone.depth_of(zone.ZONE_COLS - 1, '红') == '敌家'
    assert zone.depth_of(zone.ZONE_COLS - 1, '蓝') == '自家深'
    # 同一低x点(2,5): 红=己方侧, 蓝=敌方侧
    assert '自家' in zone.assign_zone(2.0, 5.0, B, '红')
    assert '敌' in zone.assign_zone(2.0, 5.0, B, '蓝')
    # 同一高x点(9,5): 红=敌方侧, 蓝=己方侧
    assert '敌' in zone.assign_zone(9.0, 5.0, B, '红')
    assert '自家' in zone.assign_zone(9.0, 5.0, B, '蓝')


def test_zone_neighbors_orthogonal_incl_self():
    n = zone.zone_neighbors(2, 1)
    assert (2, 1) in n and (1, 1) in n and (3, 1) in n and (2, 0) in n and (2, 2) in n
    assert (1, 0) not in n and (3, 2) not in n  # 无对角
    # 角落: 只含自身 + 右侧 + 下方
    assert set(zone.zone_neighbors(0, 0)) == {(0, 0), (1, 0), (0, 1)}


def test_unit_reachable_depth_prune():
    # 哨兵从(1,1)可回自家深(0,1)/自家场(1,0/1,2), 但不可进己半场(2,1)
    nxt = zone.movable_next('哨兵', (1, 1), '红')
    assert (0, 1) in nxt and (1, 1) in nxt and (1, 0) in nxt and (1, 2) in nxt
    assert (2, 1) not in nxt
    # 英雄/步兵(全图)可越过中线进入己半场
    assert (2, 1) in zone.movable_next('英雄', (1, 1), '红')
    assert (2, 1) in zone.movable_next('步兵3', (1, 1), '红')
    # 哨兵若出现在敌半场(3,1), 只能停留(自身) —— 不再误判"无合法状态"
    assert zone.movable_next('哨兵', (3, 1), '红') == [(3, 1)]


def test_unit_reachable_non_adjacent_false():
    # 非正交邻接一律不可达
    assert not zone.unit_reachable('英雄', (1, 1), (3, 1), '红')
    assert not zone.unit_reachable('哨兵', (1, 1), (2, 1), '红')
