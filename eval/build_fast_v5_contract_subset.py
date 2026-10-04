"""Build a small, deterministic V5 subset for fast structural-contract tuning.

The subset is not an accuracy benchmark.  It preserves the cases needed for
the output contract: red/blue, forecast/no-forecast, every global mode, and
the blue-hero-unseen augmentation that invalidated the earlier candidate.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_KEYS = ('perspective', 'variant', 'teacher_mode')


def _record_key(row, keys):
    provenance = row.get('provenance') or {}
    return tuple(provenance.get(key) for key in keys)


def _has_low_risk_no_target_anchor(row):
    """Return whether a legal completion demonstrates the risky enum boundary."""
    try:
        units = json.loads(row['completion']).get('units') or {}
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return any(action.get('target') == '无' and action.get('risk') == '低'
               for action in units.values() if isinstance(action, dict))


def build_subset(records, *, per_primary_cell=32, per_hidden_cell=16,
                 risk_anchor_count=0, seed=20260929):
    """Sample contract coverage without duplicate rows."""
    if int(per_primary_cell) <= 0 or int(per_hidden_cell) <= 0:
        raise ValueError('per-cell sample counts must be positive')
    rng = random.Random(int(seed))
    primary = defaultdict(list)
    hidden = defaultdict(list)
    for row in records:
        key = _record_key(row, PRIMARY_KEYS)
        primary[key].append(row)
        provenance = row.get('provenance') or {}
        if provenance.get('augmentation') == 'blue_hero_unseen':
            hidden[(provenance.get('variant'), provenance.get('teacher_mode'))].append(row)

    expected_primary = {(side, variant, mode)
                        for side in ('红', '蓝')
                        for variant in ('forecast', 'no_forecast')
                        for mode in ('进攻', '均衡', '防守')}
    missing_primary = sorted(expected_primary - set(primary))
    if missing_primary:
        raise ValueError(f'missing primary cells: {missing_primary}')
    expected_hidden = {(variant, mode)
                       for variant in ('forecast', 'no_forecast')
                       for mode in ('进攻', '均衡', '防守')}
    missing_hidden = sorted(expected_hidden - set(hidden))
    if missing_hidden:
        raise ValueError(f'missing blue hero unseen cells: {missing_hidden}')

    selected = []
    selected_hashes = set()
    for key in sorted(expected_primary):
        rows = list(primary[key])
        rng.shuffle(rows)
        if len(rows) < int(per_primary_cell):
            raise ValueError(f'primary cell {key} has only {len(rows)} rows')
        for row in rows[:int(per_primary_cell)]:
            digest = row['instruction'] + '\n' + row['completion']
            if digest not in selected_hashes:
                selected.append(row)
                selected_hashes.add(digest)
    for key in sorted(expected_hidden):
        rows = list(hidden[key])
        rng.shuffle(rows)
        if len(rows) < int(per_hidden_cell):
            raise ValueError(f'hidden cell {key} has only {len(rows)} rows')
        for row in rows[:int(per_hidden_cell)]:
            digest = row['instruction'] + '\n' + row['completion']
            if digest not in selected_hashes:
                selected.append(row)
                selected_hashes.add(digest)
    risk_anchor_count = int(risk_anchor_count)
    if risk_anchor_count < 0:
        raise ValueError('risk anchor count cannot be negative')
    if risk_anchor_count:
        anchors = [row for row in selected if _has_low_risk_no_target_anchor(row)]
        rng.shuffle(anchors)
        if len(anchors) < risk_anchor_count:
            raise ValueError(f'only {len(anchors)} risk anchors available')
        # Repeating a bounded subset is deliberate loss reweighting for the
        # closed risk enum, not new decision evidence or quality data.
        selected.extend(anchors[:risk_anchor_count])
    rng.shuffle(selected)
    return selected


def _counts(records):
    return {
        'primary_cells': {
            '|'.join(map(str, key)): count
            for key, count in sorted(Counter(_record_key(row, PRIMARY_KEYS) for row in records).items())
        },
        'augmentation': dict(Counter((row.get('provenance') or {}).get('augmentation', 'observed')
                                     for row in records)),
        'low_risk_no_target_anchor_rows': sum(_has_low_risk_no_target_anchor(row)
                                               for row in records),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='out_m2/temporal_visible_context_dataset_v5_observable_slots_run1/m2_dataset.jsonl')
    parser.add_argument('--out-dir', default='out_m2/temporal_visible_context_dataset_v5_contract_fast')
    parser.add_argument('--per-primary-cell', type=int, default=32)
    parser.add_argument('--per-hidden-cell', type=int, default=16)
    parser.add_argument('--risk-anchor-count', type=int, default=0,
                        help='repeat bounded legal low-risk/no-target examples for enum tuning')
    parser.add_argument('--seed', type=int, default=20260929)
    args = parser.parse_args(argv)
    source = ROOT / args.source
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    subset = build_subset(records, per_primary_cell=args.per_primary_cell,
                          per_hidden_cell=args.per_hidden_cell,
                          risk_anchor_count=args.risk_anchor_count, seed=args.seed)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / 'm2_dataset.jsonl'
    output.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                              for row in subset), encoding='utf-8')
    manifest = {
        'schema_version': 'visible-v5-fast-contract-subset-v1',
        'source': str(source), 'rows': len(subset), 'seed': int(args.seed),
        'per_primary_cell': int(args.per_primary_cell),
        'per_hidden_cell': int(args.per_hidden_cell),
        'risk_anchor_count': int(args.risk_anchor_count),
        'purpose': 'fast structural-contract tuning only; not a quality benchmark',
        **_counts(subset),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'dataset': str(output), 'manifest': str(out_dir / 'manifest.json'),
                      'rows': len(subset), **_counts(subset)}, ensure_ascii=False))


if __name__ == '__main__':
    raise SystemExit(main())
