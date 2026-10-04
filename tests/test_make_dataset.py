# -*- coding: utf-8 -*-
"""make_dataset.rebalance_records logic tests (no GPU)."""
import os, sys, json
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from finetune import make_dataset as md

ATTACK, BAL, DEF = '\u8fdb\u653b', '\u5747\u8861', '\u9632\u5b88'


def _rec(mode):
    return {'instruction': 't=0.0',
            'completion': json.dumps({'global_mode': mode}, ensure_ascii=False)}


def test_record_mode_parses_completion():
    assert md.record_mode(_rec(ATTACK)) == ATTACK
    assert md.record_mode({'completion': json.dumps({'t': 0.0})}) == BAL


def test_rebalance_upsamples_rare_mode_to_target():
    recs = ([_rec(ATTACK) for _ in range(56)]
            + [_rec(DEF) for _ in range(37)]
            + [_rec(BAL) for _ in range(7)])
    out = md.rebalance_records(recs, max_upsample=3)
    c = Counter(md.record_mode(r) for r in out)
    assert len(out) >= len(recs)
    assert c[ATTACK] >= 56 and c[DEF] >= 37 and c[BAL] >= 7
    assert c[ATTACK] == 56
    assert c[BAL] <= 7 * 3
    assert c[BAL] > 7
    assert c[ATTACK] > c[DEF]
    out2 = md.rebalance_records(recs, seed=0, max_upsample=3)
    assert [md.record_mode(r) for r in out] == [md.record_mode(r) for r in out2]


def test_rebalance_custom_ratios_and_seed_reproducible():
    recs = [_rec(m) for m in (ATTACK, BAL, BAL, DEF)]
    out = md.rebalance_records(recs, target_ratios={ATTACK: 1, BAL: 1, DEF: 1})
    c = Counter(md.record_mode(r) for r in out)
    assert len(out) >= len(recs)
    assert c[ATTACK] == 2 and c[BAL] == 2 and c[DEF] == 2


def test_rebalance_defense_majority_capped_upsample():
    recs = ([_rec(DEF) for _ in range(2526)]
            + [_rec(ATTACK) for _ in range(360)]
            + [_rec(BAL) for _ in range(474)])
    out = md.rebalance_records(recs, max_upsample=3)
    c = Counter(md.record_mode(r) for r in out)
    assert c[DEF] >= 2526 and c[ATTACK] >= 360 and c[BAL] >= 474
    assert c[ATTACK] <= 360 * 3
    assert c[BAL] <= 474 * 3
    assert c[ATTACK] / len(out) < 0.35
    assert c[ATTACK] > 360


def test_rebalance_max_upsample_one_keeps_counts():
    recs = [_rec(DEF) for _ in range(10)] + [_rec(ATTACK) for _ in range(2)]
    out = md.rebalance_records(
        recs, max_upsample=1,
        target_ratios={ATTACK: 45, DEF: 30, BAL: 25})
    c = Counter(md.record_mode(r) for r in out)
    assert c[ATTACK] == 2 and c[DEF] == 10


def test_mode_sample_weights_boost_rare_class():
    from finetune import lora_train as lt
    recs = [_rec(DEF) for _ in range(90)] + [_rec(ATTACK) for _ in range(10)]
    weights, class_w = lt._mode_sample_weights(recs)
    assert len(weights) == 100
    assert class_w[ATTACK] > class_w[DEF]
    assert abs(sum(weights) / len(weights) - 1.0) < 0.15
