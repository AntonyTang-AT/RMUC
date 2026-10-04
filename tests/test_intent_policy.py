# -*- coding: utf-8 -*-
"""intent_policy 无 torch 单测。"""
from labels import intent_policy as ip
from schema import layered as sl


def test_no_intent_unchanged():
    u = {'target': '敌方步兵', 'move': '游击', 'fire': True, 'risk': '低'}
    out = ip.apply_intent_bias(u, opp_intent=None, unit_class='哨兵')
    assert out['move'] == '游击'
    assert out['target'] == '敌方步兵'


def test_destroy_base_biases_defend():
    u = {'target': '无', 'move': '推进', 'fire': False, 'risk': '低'}
    oi = {'top': '拆基地', 'probs': {'拆基地': 0.8}}
    out = ip.apply_intent_bias(u, opp_intent=oi, unit_class='英雄', global_mode='均衡')
    assert out['move'] == '回防'
    assert out['target'] == '敌方基地'
    assert out['fire'] is True
    assert out['target'] in sl.TARGETS
    assert out['move'] in sl.MOVES


def test_watch_sentry_sentry_class_reroute():
    u = {'target': '敌方前哨站', 'move': '推进', 'fire': True, 'risk': '中'}
    oi = {'top': '盯哨兵', 'probs': {'盯哨兵': 0.7}}
    out = ip.apply_intent_bias(u, opp_intent=oi, unit_class='哨兵')
    assert out['move'] == '绕行'
    assert out['target'] == '敌方哨兵'


def test_defend_mode_blocks_push_bias():
    u = {'target': '无', 'move': '回防', 'fire': False, 'risk': '低'}
    oi = {'top': '换血前排', 'probs': {'换血前排': 0.9}}
    out = ip.apply_intent_bias(u, opp_intent=oi, unit_class='英雄', global_mode='防守')
    assert out['move'] == '回防'  # 防守下不强制推进


def test_bias_units_dict_covers_classes():
    units = {
        '英雄': {'target': '无', 'move': '无', 'fire': False, 'risk': '低'},
        '哨兵': {'target': '无', 'move': '无', 'fire': False, 'risk': '低'},
    }
    oi = {'top': '压前哨', 'probs': {'压前哨': 0.6}}
    out = ip.bias_units_dict(units, opp_intent=oi, global_mode='均衡')
    assert '英雄' in out and '哨兵' in out
    assert out['哨兵']['move'] == '回防'
