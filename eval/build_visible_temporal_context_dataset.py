"""Rebuild a QLoRA dataset with only radar/sensor-visible LLM state.

The existing temporal-context dataset supplies frozen causal forecast sidecars
and teacher completions.  Its original state digest predates the strict
visibility prompt contract, so this builder keeps the teacher completion and
forecast sidecar while replacing only the state portion with a validated
side-specific sensor/radar projection at the same decision time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _forecast_context(instruction):
    marker = '预测参考信息'
    start = str(instruction).find(marker)
    if start < 0:
        raise ValueError('source instruction has no forecast context')
    opening = str(instruction).find('{', start)
    if opening < 0:
        raise ValueError('forecast context has no JSON object')
    try:
        value, _end = json.JSONDecoder().raw_decode(str(instruction)[opening:])
    except json.JSONDecodeError as exc:
        raise ValueError('forecast context JSON is invalid') from exc
    if not isinstance(value, dict) or value.get('schema_version') != 'forecast-context-v1':
        raise ValueError('unexpected forecast context schema')
    return value


def _issue_time(instruction):
    match = re.search(r'局势:\s*t=([0-9]+(?:\.[0-9]+)?)', str(instruction))
    if not match:
        raise ValueError('source instruction has no issue timestamp')
    return float(match.group(1))


def _group_source_rows(rows, game_ids):
    """Partition the source in its manifest's game order; fail on drift."""
    groups, current = [], []
    last_t = None
    for row in rows:
        t = _issue_time(row.get('instruction', ''))
        if last_t is not None and t <= last_t:
            if not current:
                raise ValueError('empty source game partition')
            groups.append(current)
            current = []
        current.append(row)
        last_t = t
    if current:
        groups.append(current)
    if len(groups) != len(game_ids):
        raise ValueError(f'source partitions={len(groups)} but manifest games={len(game_ids)}')
    return dict(zip(game_ids, groups))


def build_dataset(source_rows, game_ids, *, side='红', max_chars=760):
    import run_m2
    from eval.llm_visible_input import (build_visible_llm_input,
                                        format_visible_training_instruction,
                                        validate_visible_llm_input)

    by_game = _group_source_rows(source_rows, game_ids)
    output, input_hashes = [], []
    for game_id in game_ids:
        dps, _bounds = run_m2.load_decision_points(int(game_id))
        source_group = by_game[game_id]
        if len(dps) != len(source_group):
            raise ValueError(f'game {game_id}: decision points={len(dps)} source rows={len(source_group)}')
        snapshots = run_m2.load_observable_snapshots(int(game_id), team=side)
        for index, (dp, source) in enumerate(zip(dps, source_group)):
            issued_at = float(dp.get('t', 0.0))
            if abs(issued_at - _issue_time(source['instruction'])) > 1e-6:
                raise ValueError(f'game {game_id} row {index}: timestamp mismatch')
            visible = build_visible_llm_input(
                snapshots, game_id=int(game_id), side=side,
                issued_at_s=issued_at, history_frames=4)
            if not validate_visible_llm_input(visible):
                raise ValueError(f'game {game_id} row {index}: visibility audit failed')
            context = _forecast_context(source['instruction'])
            instruction = format_visible_training_instruction(
                visible, forecast_context=context, max_chars=max_chars)
            completion = source.get('completion')
            if not isinstance(completion, str) or not completion:
                raise ValueError(f'game {game_id} row {index}: missing completion')
            output.append({
                'instruction': instruction,
                'completion': completion,
                'provenance': {
                    'game_id': int(game_id), 'issued_at_s': issued_at,
                    'perspective': side,
                    'visibility_policy': visible['visibility_policy'],
                    'visible_input_sha256': visible['input_sha256'],
                    'forecast_context_sha256': hashlib.sha256(
                        json.dumps(context, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':')).encode('utf-8')).hexdigest(),
                },
            })
            input_hashes.append(visible['input_sha256'])
    return output, input_hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='out_m2/temporal_context_dataset_promptfix_full/m2_dataset.jsonl')
    parser.add_argument('--source-manifest', default='out_m2/temporal_context_dataset_promptfix_full/manifest.json')
    parser.add_argument('--out-dir', default='out_m2/temporal_visible_context_dataset')
    parser.add_argument('--side', default='红', choices=('红', '蓝'))
    parser.add_argument('--max-chars', type=int, default=760)
    args = parser.parse_args(argv)

    source_rows = _read_jsonl(ROOT / args.source)
    source_manifest = json.loads((ROOT / args.source_manifest).read_text(encoding='utf-8'))
    game_ids = [int(value) for value in source_manifest['game_ids']]
    records, input_hashes = build_dataset(
        source_rows, game_ids, side=args.side, max_chars=args.max_chars)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / 'm2_dataset.jsonl'
    with output_path.open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    manifest = {
        'schema_version': 'visible-temporal-context-dataset-v1',
        'source_dataset': str(args.source),
        'source_manifest': str(args.source_manifest),
        'rows': len(records),
        'game_ids': game_ids,
        'perspective': args.side,
        'visibility_policy': 'rmuc_2026_referee_radar_protocol_v1',
        'all_visible_inputs_validated': True,
        'forecast_context_source': 'frozen causal forecast-context-v1 from source rows',
        'prompt_max_chars': int(args.max_chars),
        'visible_input_set_sha256': hashlib.sha256(
            '\n'.join(input_hashes).encode('utf-8')).hexdigest(),
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'dataset': str(output_path), 'manifest': str(out_dir / 'manifest.json'),
                      'rows': len(records), 'games': len(game_ids)}, ensure_ascii=False))


if __name__ == '__main__':
    raise SystemExit(main())
