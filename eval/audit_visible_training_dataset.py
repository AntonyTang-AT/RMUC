"""Audit the paired red/blue visible-context LoRA training dataset."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema.layered import validate
from eval.llm_visible_input import conservative_slot_action


DEFAULT_HELDOUT_GAMES = (
    1778661259223,
    1778661970735,
    1778663517697,
    1778664469370,
)


def audit(dataset_path, manifest_path, *, heldout_games=DEFAULT_HELDOUT_GAMES,
          heldout_manifests=()):
    rows = [json.loads(line) for line in Path(dataset_path).read_text(encoding='utf-8').splitlines()
            if line.strip()]
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    groups = defaultdict(set)
    sides, variants, modes, source_games = Counter(), Counter(), Counter(), set()
    invalid = []
    bad_context = []
    bad_slot_contract = []
    requires_observable_slots = str(manifest.get('schema_version', '')).startswith(
        'visible-temporal-context-dataset-v5')
    for row in rows:
        provenance = row.get('provenance') or {}
        key = (provenance.get('game_id'), provenance.get('issued_at_s'),
               provenance.get('perspective'))
        variant = provenance.get('variant')
        groups[key].add(variant)
        sides[provenance.get('perspective')] += 1
        variants[variant] += 1
        modes[provenance.get('teacher_mode')] += 1
        source_games.add(provenance.get('game_id'))
        try:
            legal = bool(validate(json.loads(row['completion']))[0])
        except (KeyError, TypeError, json.JSONDecodeError):
            legal = False
        if not legal:
            invalid.append(key)
        has_context = bool(provenance.get('forecast_context_sha256'))
        if (variant == 'forecast') != has_context:
            bad_context.append(key)
        if requires_observable_slots:
            statuses = provenance.get('slot_statuses') or {}
            if set(statuses) != {'英雄', '工程', '步兵3', '步兵4', '空中', '哨兵'}:
                bad_slot_contract.append(key)
            else:
                try:
                    completion = json.loads(row['completion'])
                    for unit_type, status in statuses.items():
                        if status == 'current':
                            continue
                        if completion['units'].get(unit_type) != conservative_slot_action(status):
                            bad_slot_contract.append(key)
                            break
                except (KeyError, TypeError, json.JSONDecodeError):
                    bad_slot_contract.append(key)
    expected_variants = {'forecast', 'no_forecast'}
    bad_groups = [key for key, group in groups.items() if group != expected_variants]
    heldout_ids = set(map(int, heldout_games))
    heldout_sources = []
    for path in heldout_manifests:
        heldout_manifest = json.loads(Path(path).read_text(encoding='utf-8'))
        ids = set(map(int, heldout_manifest.get('game_ids') or []))
        heldout_ids.update(ids)
        heldout_sources.append({
            'path': str(Path(path)),
            'games': len(ids),
        })
    heldout_overlap = sorted(heldout_ids.intersection(map(int, source_games)))
    result = {
        'schema_version': 'visible-training-dataset-audit-v1',
        'rows': len(rows), 'manifest_rows': manifest.get('rows'),
        'unique_side_time_groups': len(groups),
        'bad_variant_groups': len(bad_groups),
        'invalid_completions': len(invalid),
        'bad_forecast_provenance': len(bad_context),
        'bad_observable_slot_contract': len(bad_slot_contract),
        'heldout_overlap': heldout_overlap,
        'heldout_games_checked': len(heldout_ids),
        'heldout_manifests': heldout_sources,
        'games': len(source_games), 'side_counts': dict(sides),
        'variant_counts': dict(variants), 'teacher_mode_counts': dict(modes),
        'passed': (len(rows) == int(manifest.get('rows', -1)) and not bad_groups
                   and not invalid and not bad_context and not bad_slot_contract
                   and not heldout_overlap),
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--heldout-manifests', nargs='*', default=())
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = audit(args.dataset, args.manifest,
                   heldout_manifests=args.heldout_manifests)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
