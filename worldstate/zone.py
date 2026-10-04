# -*- coding: utf-8 -*-
"""世界场地功能分区 + 单位可达性(状态转移规则)。

设计要点(对应 Q1/Q3):
- 地块按"功能"而非均匀网格: 纵向 x 切成 6 条"深度带"(从己方基地一路到敌方基地),
  横向 y 切成 3 条"道"(上翼/中路/下翼) => 6×3 = 18 个语义区块(比旧 4×3=12 更多、按功能)。
- 深度带名是**队伍相对**的(以"己方基地"为原点): 红方主场=低 x, 蓝方主场=高 x
  (与 labels/sentry_labels._enemy_in_half 的约定一致); 蓝方的区块名按 x 轴镜像。
- 状态转移约束(Q1): 单位下一区块只能 = **当前区块或正交邻区**(zone_neighbors);
  再按单位机动性先验 prune(unit_reachable)—— 哨兵/工程受限, 兵种/英雄/空中全图。
- 数据局限(诚实标注): 建筑(基地/前哨)在 timeseries 里坐标恒为 (0,0) 占位, 无法用真实地标
  锚定; 故深度用"场地归一化坐标 + 已知中线"而非建筑坐标。field_bounds 仍来自全场机器人。

★ 这是 replay/near-bandit 下的"状态空间描述"与"转移过滤", 不是前向动力学(那需要路线 B)。
"""

# 纵向 x 带数 / 横向 y 道数
ZONE_COLS = 6
ZONE_ROWS = 3

# 深度带名: 从"己方基地"(自家深) 到 "敌方基地"(敌家), 越靠敌家越"推进"。
DEPTH_FROM_HOME = ['自家深', '自家场', '己半场', '敌半场', '敌场', '敌家']
# 横向道名(绝对 y): 上翼 / 中路 / 下翼
LANE_NAMES = ['上翼', '中路', '下翼']

# 单位 -> 可达深度带集合(None=全图)。RoboMaster 单位机动性先验(无真实地标, 用规则近似)。
#   哨兵 = 基地炮台/轨道, 只在自家基地圈;  工程 = 防守, 不跨中线; 其余 = 全图。
MOVABLE_DEPTH = {
    '哨兵': {'自家深', '自家场'},
    '工程': {'自家深', '自家场', '己半场'},
    '英雄': None,
    '步兵3': None,
    '步兵4': None,
    '空中': None,
}
ALL_UNITS = ('英雄', '工程', '步兵3', '步兵4', '空中', '哨兵')


def zone_index(x, y, bounds):
    """(x,y) -> (ci,ri), 均 0-based, 钳到 [0,ZONE_COLS-1]/[0,ZONE_ROWS-1]。"""
    span_x = bounds['xmax'] - bounds['xmin']
    span_y = bounds['ymax'] - bounds['ymin']
    span_x = span_x if span_x > 1e-9 else 1e-9
    span_y = span_y if span_y > 1e-9 else 1e-9
    ci = int((x - bounds['xmin']) / span_x * ZONE_COLS)
    ri = int((y - bounds['ymin']) / span_y * ZONE_ROWS)
    return (max(0, min(ci, ZONE_COLS - 1)), max(0, min(ri, ZONE_ROWS - 1)))


def depth_of(ci, team):
    """列 -> 队伍相对深度带名(越靠敌家越"推进")。红方主场=低x, 蓝方=高x(镜像)。"""
    if team == '蓝':
        ci = ZONE_COLS - 1 - ci
    return DEPTH_FROM_HOME[ci]


def lane_of(ri):
    """行 -> 道名。"""
    return LANE_NAMES[ri]


def zone_label(ci, ri, team):
    """语义区块名。"""
    return f"{depth_of(ci, team)}_{lane_of(ri)}"


def assign_zone(x, y, bounds, team):
    ci, ri = zone_index(x, y, bounds)
    return zone_label(ci, ri, team)


def depth_from_xy(x, y, bounds, team):
    ci, ri = zone_index(x, y, bounds)
    return depth_of(ci, team)


# 全部可取值(红/蓝都产生同一 18 个语义名; 并集保证 assign_zone 结果一定落在其中)
ZONE_LABELS = sorted({zone_label(ci, ri, t)
                      for ci in range(ZONE_COLS) for ri in range(ZONE_ROWS)
                      for t in ('红', '蓝')})


def zone_neighbors(ci, ri):
    """当前区的正交邻区(含自身): Q1 的"只能停留或移到相邻区" —— 邻接图的边。"""
    out = [(ci, ri)]
    for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nc, nr = ci + dc, ri + dr
        if 0 <= nc < ZONE_COLS and 0 <= nr < ZONE_ROWS:
            out.append((nc, nr))
    return out


def unit_reachable(unit, from_pos, to_pos, team):
    """to_pos 是否可从 from_pos 到达: 必须正交邻接; 停留(==from)恒真; 移动则按单位深度带裁剪。"""
    ci, ri = from_pos
    if to_pos not in zone_neighbors(ci, ri):
        return False
    if to_pos == from_pos:
        return True  # 总是可停留
    allowed = MOVABLE_DEPTH.get(unit)
    if allowed is None:
        return True
    tci, _ = to_pos
    return depth_of(tci, team) in allowed


def movable_next(unit, from_pos, team):
    """单位从 from_pos 的合法下一状态集[含自身] —— 带单位约束的状态转移规则。"""
    return [p for p in zone_neighbors(*from_pos) if unit_reachable(unit, from_pos, p, team)]

# New map-aligned functional-zone API. The legacy grid symbols above remain
# available for old checkpoints; new worldstate output uses these functions.
from worldstate.functional_zone import (  # noqa: E402
    FUNCTIONAL_ZONE_LABELS,
    assign_functional_zone,
    assign_functional_zone_normalized,
    functional_zone_neighbors,
    candidate_moves as functional_candidate_moves,
    can_move_between as functional_can_move_between,
)
