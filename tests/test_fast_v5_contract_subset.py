from eval.build_fast_v5_contract_subset import build_subset


def _row(side, variant, mode, augmentation='observed', index=0):
    return {
        'instruction': f'{side}/{variant}/{mode}/{augmentation}/{index}',
        'completion': f'{{"index":{index}}}',
        'provenance': {
            'perspective': side, 'variant': variant, 'teacher_mode': mode,
            'augmentation': augmentation, 'visible_input_sha256': str(index),
        },
    }


def test_fast_subset_preserves_all_contract_cells_and_blue_hidden_cases():
    rows = []
    index = 0
    for side in ('红', '蓝'):
        for variant in ('forecast', 'no_forecast'):
            for mode in ('进攻', '均衡', '防守'):
                rows.append(_row(side, variant, mode, index=index))
                index += 1
                if side == '蓝':
                    rows.append(_row(side, variant, mode, 'blue_hero_unseen', index))
                    index += 1
    subset = build_subset(rows, per_primary_cell=1, per_hidden_cell=1, seed=1)
    primary = {(r['provenance']['perspective'], r['provenance']['variant'],
                r['provenance']['teacher_mode']) for r in subset}
    hidden = {(r['provenance']['variant'], r['provenance']['teacher_mode']) for r in subset
              if r['provenance']['augmentation'] == 'blue_hero_unseen'}
    assert len(primary) == 12
    assert len(hidden) == 6


def test_fast_subset_can_repeat_legal_low_risk_no_target_anchors():
    rows = []
    index = 0
    completion = ('{"units":{"英雄":{"target":"无","move":"无",'
                  '"fire":false,"risk":"低"}}}')
    for side in ('红', '蓝'):
        for variant in ('forecast', 'no_forecast'):
            for mode in ('进攻', '均衡', '防守'):
                row = _row(side, variant, mode, index=index)
                row['completion'] = completion
                rows.append(row)
                index += 1
                if side == '蓝':
                    hidden = _row(side, variant, mode, 'blue_hero_unseen', index)
                    hidden['completion'] = completion
                    rows.append(hidden)
                    index += 1
    baseline = build_subset(rows, per_primary_cell=1, per_hidden_cell=1, seed=1)
    subset = build_subset(rows, per_primary_cell=1, per_hidden_cell=1,
                          risk_anchor_count=3, seed=1)
    assert len(subset) == len(baseline) + 3
