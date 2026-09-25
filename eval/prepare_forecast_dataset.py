"""Extract causal, radar-masked, both-side forecast examples from SQLite."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import json
import os
import sqlite3
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from config import DB_PATH
from data.forecast_dataset import build_game_forecast_examples, dataset_metadata
from data.forecast_splits import chronological_group_split
from data.radar_visibility_replay import PROVIDER_ID
from eval.forecast_report import build_forecast_manifest


def _json_default(value):
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


def prepare_dataset(db_path=DB_PATH, out_dir=None, *, train_fraction=.6,
                    validation_fraction=.2, step=3, horizons=(10, 30, 60, 120)):
    out_dir = out_dir or os.path.join(REPO, 'out_m2', 'forecast_dataset')
    os.makedirs(out_dir, exist_ok=True)
    con = sqlite3.connect(f'file:{os.path.abspath(db_path)}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        match_rows = [dict(row) for row in con.execute('SELECT * FROM matches').fetchall()]
        split = chronological_group_split(
            match_rows, train_fraction, validation_fraction,
            sensor_provider_version=PROVIDER_ID, horizons=tuple(horizons),
        )
        # Use the chronological order fixed by the split manifest, including
        # games from the same match group in competition order.
        order = {game_id: index for index, game_id in enumerate(split['manifest']['all_game_ids'])}
        manifest_rows = []
        for row in match_rows:
            item = dict(row)
            item['competition_order'] = order[item['game_id']]
            manifest_rows.append(item)
        match_split = {name: split[name] for name in ('train', 'validation', 'test')}
        manifest = build_forecast_manifest(
            manifest_rows, match_split, horizons, PROVIDER_ID,
            label_version='global-truth-exact-time-v1',
            target_builder_version='data.forecast_dataset.v1',
        )
        manifest['chronological_split'] = split['manifest']
        manifest['dataset'] = dataset_metadata()
        match_to_split = {match_id: name for name in ('train', 'validation', 'test')
                          for match_id in split[name]}
        from data.forecast_splits import _match_id
        game_to_split = {row['game_id']: match_to_split[_match_id(row)] for row in match_rows}
        counts = {name: Counter() for name in ('train', 'validation', 'test')}
        paths = {name: os.path.join(out_dir, f'{name}.jsonl.gz')
                 for name in ('train', 'validation', 'test')}
        with ExitStack() as stack:
            writers = {name: stack.enter_context(
                gzip.open(path, 'wt', encoding='utf-8', newline='\n'))
                       for name, path in paths.items()}
            for game_id in split['manifest']['all_game_ids']:
                rows = con.execute('SELECT * FROM timeseries WHERE game_id=? ORDER BY "时刻秒"',
                                   (game_id,)).fetchall()
                game_examples = build_game_forecast_examples(
                    rows, game_id, horizons=tuple(horizons), step=step,
                )
                split_name = game_to_split[game_id]
                for side in ('红', '蓝'):
                    for example in game_examples[side]:
                        example['split'] = split_name
                        example['visibility_provider'] = PROVIDER_ID
                        writers[split_name].write(json.dumps(
                            example, ensure_ascii=False, separators=(',', ':'),
                            default=_json_default,
                        ) + '\n')
                        stats = counts[split_name]
                        stats['examples'] += 1
                        stats[f'{side}_examples'] += 1
                        stats['opponent_tracks'] += len(
                            example['observable']['opponent_entities']
                        )
                        stats['opponent_positions'] += sum(
                            'position' in entity
                            for entity in example['observable']['opponent_entities'].values()
                        )
                        for horizon, target in example['targets'].items():
                            stats['targets'] += 1
                            stats['censored_targets' if target is None else 'mature_targets'] += 1
        manifest['outputs'] = {
            name: {'path': os.path.abspath(paths[name]), **dict(counts[name])}
            for name in ('train', 'validation', 'test')
        }
        manifest_path = os.path.join(out_dir, 'manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
        return manifest
    finally:
        con.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default=DB_PATH)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--train-fraction', type=float, default=.6)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    parser.add_argument('--step', type=int, default=3)
    parser.add_argument('--horizons', type=int, nargs='+', default=[10, 30, 60, 120])
    args = parser.parse_args(argv)
    report = prepare_dataset(
        args.db, args.out_dir,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        step=args.step,
        horizons=tuple(args.horizons),
    )
    print(json.dumps({'manifest': report['manifest_sha256'], 'outputs': report['outputs']},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
