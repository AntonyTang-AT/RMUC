"""Run a complete visible-input LLM candidate evaluation matrix.

The matrix always covers red/blue x forecast/no-forecast. It writes immutable
rows, structural reports, an HDQ/ORP scorecard, and an acceptance audit under
a new output directory. No evaluation prompt receives replay-hidden or future
fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_manifest_games(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return [int(value) for value in payload['game_ids']]


def _write_failure_audit(rows_path, output_path):
    """Persist parse diagnostics next to the immutable generation rows."""
    from eval import audit_visible_lora_failures as failures

    rows = [json.loads(line) for line in Path(rows_path).read_text(encoding='utf-8').splitlines()
            if line.strip()]
    report = failures.audit(rows)
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def run_matrix(*, adapter, games, output_dir, device='cuda', checkpoint='auto',
               limit=12, sampling='uniform', batch_size=4, max_new=200,
               prompt_max_length=896, instruction_max_chars=640,
               min_games=12, min_mode_rows=12, min_mode_games=12):
    from eval import audit_llm_candidate_acceptance as acceptance
    from eval import decision_quality_scorecard as quality
    from eval import evaluate_visible_lora_context as visible
    from config import DB_PATH

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    game_rows = visible._load_games(games, limit, sampling=sampling)
    reports, row_paths, failure_audit_paths = [], [], []
    for side in ('红', '蓝'):
        for use_forecast in (True, False):
            label = f"{('red' if side == '红' else 'blue')}_{'forecast' if use_forecast else 'noforecast'}"
            rows_output = output_dir / f'{label}.rows.jsonl'
            result = visible.evaluate(
                adapter, game_rows, side=side, device=device, checkpoint=checkpoint,
                use_forecast=use_forecast, include_region10=True, max_new=max_new,
                prompt_max_length=prompt_max_length, batch_size=batch_size,
                instruction_max_chars=instruction_max_chars, rows_output=str(rows_output))
            report = {
                'schema_version': 'visible-lora-forecast-context-eval-v1',
                'games': [game_id for game_id, _dps, _bounds in game_rows],
                'limit_per_game': int(limit), 'sampling_policy': sampling,
                'sampled_issued_at_s': {
                    str(game_id): [float(dp.get('t', 0.0)) for dp in dps]
                    for game_id, dps, _bounds in game_rows
                },
                'result': result,
            }
            report_path = output_dir / f'{label}.report.json'
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            failure_audit_path = output_dir / f'{label}.failure_audit.json'
            _write_failure_audit(rows_output, failure_audit_path)
            reports.append(report)
            row_paths.append(rows_output)
            failure_audit_paths.append(failure_audit_path)
    rows = []
    for path in row_paths:
        rows.extend(quality._read_jsonl(path))
    scorecard = quality.score(rows, DB_PATH, lookahead_delta=5)
    quality_path = output_dir / 'decision_quality.json'
    quality_path.write_text(json.dumps(scorecard, ensure_ascii=False, indent=2), encoding='utf-8')
    audit = acceptance.audit(reports, scorecard, min_games=min_games,
                             min_mode_rows=min_mode_rows, min_mode_games=min_mode_games)
    audit_path = output_dir / 'acceptance.json'
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {
        'schema_version': 'llm-candidate-matrix-v1',
        'adapter': str(Path(adapter).resolve()),
        'games': [game_id for game_id, _dps, _bounds in game_rows],
        'limit_per_game': int(limit), 'sampling_policy': sampling,
        'required_scenarios': [
            {'side': side, 'forecast_context_enabled': forecast}
            for side in ('红', '蓝') for forecast in (True, False)
        ],
        'rows': [str(path.resolve()) for path in row_paths],
        'failure_audits': [str(path.resolve()) for path in failure_audit_paths],
        'reports': [str((output_dir / f'{label}.report.json').resolve())
                    for label in ('red_forecast', 'red_noforecast',
                                  'blue_forecast', 'blue_noforecast')],
        'quality': str(quality_path.resolve()),
        'acceptance': str(audit_path.resolve()),
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'manifest': manifest, 'acceptance': audit}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--games', nargs='*', type=int, default=None)
    parser.add_argument('--games-manifest', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--temporal-checkpoint', default='auto')
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--sampling', choices=('uniform', 'head'), default='uniform')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-new', type=int, default=200)
    parser.add_argument('--prompt-max-length', type=int, default=896)
    parser.add_argument('--instruction-max-chars', type=int, default=640)
    parser.add_argument('--min-games', type=int, default=12)
    parser.add_argument('--min-mode-rows', type=int, default=12)
    parser.add_argument('--min-mode-games', type=int, default=12)
    args = parser.parse_args(argv)
    if bool(args.games) == bool(args.games_manifest):
        parser.error('provide exactly one of --games or --games-manifest')
    games = args.games if args.games else _load_manifest_games(args.games_manifest)
    result = run_matrix(
        adapter=args.adapter, games=games, output_dir=args.output_dir, device=args.device,
        checkpoint=args.temporal_checkpoint, limit=args.limit, sampling=args.sampling,
        batch_size=args.batch_size, max_new=args.max_new,
        prompt_max_length=args.prompt_max_length,
        instruction_max_chars=args.instruction_max_chars, min_games=args.min_games,
        min_mode_rows=args.min_mode_rows, min_mode_games=args.min_mode_games)
    print(json.dumps({'output_dir': str(Path(args.output_dir).resolve()),
                      'gates': result['acceptance']['gates']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
