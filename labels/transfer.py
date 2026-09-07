# -*- coding: utf-8 -*-
"""M2 跨单元迁移: 覆盖率低的类复用源类(站位/结构相近)的 (move, fire), 生成增广训练对。
纯 numpy, 无 torch。"""
import copy
from schema import layered as sl

# 类 -> 源类(站位相近): 后排慢速(哨兵/工程/空中) 互相增广; 前排(步兵3/步兵4) 互相增广; 英雄回退空中。
TRANSFER_SOURCE = {'哨兵': '工程', '工程': '哨兵', '空中': '工程',
                   '步兵4': '步兵3', '步兵3': '步兵4', '英雄': '空中'}


def augment_transfer(decisions, coverage, source_policy=None, threshold=0.5):
    """coverage: {cls: 可达样本比例}; 低于 threshold 的类用源类样本增广。
    source_policy 为 None -> 直接取源类在本决策的 (move, fire) 复制到该类。
    返回 (augmented_decisions, n_added)。"""
    augmented = []
    n_added = 0
    for d in decisions:
        d2 = copy.deepcopy(d)
        for cls, src in TRANSFER_SOURCE.items():
            cov = coverage.get(cls, 1.0)
            if cov < threshold and src in d2['units'] and d2['units'][src]['move'] != '无':
                d2['units'][cls]['move'] = d2['units'][src]['move']
                d2['units'][cls]['fire'] = d2['units'][src]['fire']
                n_added += 1
        augmented.append(d2)
    return augmented, n_added


def default_coverage(decisions):
    """默认覆盖率: 某类在所有决策中 'move != 无'(有实际动作) 的比例。"""
    out = {}
    for cls in sl.UNIT_CLASSES:
        n = sum(1 for d in decisions if d['units'].get(cls, {}).get('move', '无') != '无')
        out[cls] = n / len(decisions) if decisions else 0.0
    return out
