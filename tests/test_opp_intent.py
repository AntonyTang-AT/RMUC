# -*- coding: utf-8 -*-
"""敌方意图概率模块单测（无 torch）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from labels import opp_intent as oi


def _w(t=10.0, our_base=1.0, opp_base=1.0, our_out=1.0, opp_out=1.0,
       our_sentry_hp=1.0, opp_sentry_xy=(0.8, 0.5), eng_mid=False):
    robots = {
        'r1': {'team': '红', 'type': '哨兵', 'hp_ratio': our_sentry_hp,
               'norm_xy': [0.2, 0.5], 'dfire': 0},
        'r2': {'team': '红', 'type': '英雄', 'hp_ratio': 0.9, 'norm_xy': [0.3, 0.5], 'dfire': 0},
        'b1': {'team': '蓝', 'type': '哨兵', 'hp_ratio': 1.0,
               'norm_xy': list(opp_sentry_xy), 'dfire': 0},
        'b2': {'team': '蓝', 'type': '英雄', 'hp_ratio': 0.9, 'norm_xy': [0.7, 0.5], 'dfire': 0},
    }
    if eng_mid:
        robots['b3'] = {'team': '蓝', 'type': '工程', 'hp_ratio': 1.0,
                        'norm_xy': [0.5, 0.5], 'dfire': 0}
    buildings = {
        'rb': {'team': '红', 'type': '基地', 'hp_ratio': our_base},
        'bb': {'team': '蓝', 'type': '基地', 'hp_ratio': opp_base},
        'ro': {'team': '红', 'type': '前哨站', 'hp_ratio': our_out},
        'bo': {'team': '蓝', 'type': '前哨站', 'hp_ratio': opp_out},
    }
    return {'t': t, 'robots': robots, 'buildings': buildings, 'team': {}}


def test_probs_sum_to_one_and_cover_intents():
    out = oi.infer_opponent_intent(_w())
    assert set(out['probs']) == set(oi.INTENTS)
    assert abs(sum(out['probs'].values()) - 1.0) < 1e-3
    assert out['top'] in oi.INTENTS
    assert out['source'] == 'delta_s+game_boost'


def test_outpost_damage_raises_press_outpost():
    prev = _w(opp_out=1.0)
    curr = _w(opp_out=0.7)
    out = oi.infer_opponent_intent(curr, prev)
    assert out['probs']['压前哨'] > out['probs']['摸资源/中立']


def test_our_sentry_damage_raises_focus_sentry():
    prev = _w(our_sentry_hp=1.0, opp_sentry_xy=(0.25, 0.5))
    curr = _w(our_sentry_hp=0.7, opp_sentry_xy=(0.22, 0.5))
    out = oi.infer_opponent_intent(curr, prev)
    assert out['probs']['盯哨兵'] >= 0.15


def test_game_boost_lifts_repeated_top():
    tracker = oi.GameIntentTracker()
    w = _w(opp_out=0.5)
    w_prev = _w(opp_out=0.9)
    first = oi.infer_opponent_intent(w, w_prev, tracker=tracker)
    # 多次观测同一局势，抬升应对 top 有帮助
    for _ in range(5):
        oi.infer_opponent_intent(w, w_prev, tracker=tracker)
    later = oi.infer_opponent_intent(w, w_prev, tracker=tracker)
    top = first['top']
    assert later['probs'][top] >= first['probs'][top] - 1e-6
    assert sum(tracker.counts.values()) >= 6


def test_format_intent_line():
    out = oi.infer_opponent_intent(_w())
    line = oi.format_intent_line(out)
    assert '敌方意图' in line
    assert out['top'] in line


def test_annotate_series_length():
    ws = [_w(t=float(i), opp_out=1.0 - i * 0.05) for i in range(4)]
    outs = oi.annotate_series(ws)
    assert len(outs) == 4
    assert all(abs(sum(o['probs'].values()) - 1.0) < 1e-3 for o in outs)
