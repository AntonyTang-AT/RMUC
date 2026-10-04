"""Collect reproducible experiment artifacts into one acceptance summary."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _payload_from_html(path):
    text = Path(path).read_text(encoding='utf-8')
    start = text.index('const DATA=') + len('const DATA=')
    end = text.index(';\nconst root', start)
    return json.loads(text[start:end])


def summarize(model_report, replay_html, paired_probe):
    report = _read_json(model_report)
    payload = _payload_from_html(replay_html)
    paired = _read_json(paired_probe)
    accuracy = report.get('accuracy_by_task_horizon', {})
    entity = report.get('entity_forecast_metrics', {})
    def task_accuracy(model, task):
        return {key: value for key, value in accuracy.get(model, {}).items()
                if key.startswith(task + '@')}
    return {
        'schema': 'rm-temporal-experiment-summary-v1',
        'model_evaluation': {
            'report': str(model_report),
            'scored_count': report.get('scored_count'),
            'red_blue_complete': report.get('red_blue_complete'),
            'weighted_acceptance': report.get('weighted_acceptance'),
            'accuracy': {
                'temporal_gru_v1': {
                    'global_mode': task_accuracy('temporal_gru_v1', 'global_mode'),
                    'opponent_trend': task_accuracy('temporal_gru_v1', 'opponent_trend'),
                    'enemy_region': task_accuracy('temporal_gru_v1', 'enemy_region'),
                },
                'persistence_v1': {
                    'global_mode': task_accuracy('persistence_v1', 'global_mode'),
                    'opponent_trend': task_accuracy('persistence_v1', 'opponent_trend'),
                    'enemy_region': task_accuracy('persistence_v1', 'enemy_region'),
                },
            },
            'entity_metrics': {
                horizon: {
                    'presence_accuracy': values.get('presence_accuracy'),
                    'position_mae_normalized': values.get('position_mae_normalized'),
                    'functional_zone_accuracy': values.get('functional_zone_accuracy'),
                    'transition_legal_rate': values.get('transition_legal_rate'),
                }
                for horizon, values in entity.items()
            },
        },
        'temporal_consistency': payload.get('temporal_consistency'),
        'adaptive_calibration': payload.get('adaptive_calibration'),
        'qwen_open_jev_paired_probe': paired,
        'limitations': [
            'The replay decision grid is 3 seconds; the requested t+20 comparison uses nearest t+21 and reports the 1 second quantisation error.',
            'The current Qwen/Open-Jev probe contains 8 paired rows and is diagnostic_only; a ranking requires at least 128 paired rows and a running JeV endpoint.',
            'Persistence can outperform the learned model on short stable labels; the production route therefore keeps task/horizon-specific adaptive switching enabled.',
        ],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'out_m2' / 'experiments' / 'experiment_summary_20260926.json'
    summary = summarize(
        root / 'out_m2' / 'temporal_forecast_weighted_v3' / 'temporal_forecast_report.json',
        Path(r'C:\Users\AntonyTang\.codex\visualizations\2026\09\24\01a0d26c-e553-7562-9bc2-455ceecdbfaa\rm-battle-replay.html'),
        root / 'out_m2' / 'temporal_forecast' / 'qwen_jev_paired_comparison.json',
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output),
                      'scored_count': summary['model_evaluation']['scored_count'],
                      'consistency_pairs': (summary['temporal_consistency'] or {}).get('pairs')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

