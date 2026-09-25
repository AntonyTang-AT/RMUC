"""Forecast run manifests and fail-closed deployment readiness summaries."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


SIDES = ('红', '蓝')
DEFAULT_HORIZONS = (10, 30, 60, 120)


def _digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), default=str).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _match_id(row):
    match_id = row.get('match_id')
    if match_id is not None and str(match_id).strip():
        return str(match_id)
    region = row.get('赛区', row.get('region'))
    number = row.get('场次号', row.get('match_number'))
    if region is None or number is None or not str(region).strip():
        raise ValueError('every match row must include match_id or 赛区 plus 场次号')
    return f'{region}:{number}'


def build_forecast_manifest(matches, split, horizons=DEFAULT_HORIZONS,
                            visibility_provider=None, label_version='v1', *,
                            region_map_version=None, target_builder_version='v1',
                            feature_cutoff='observed_at <= issued_at and received_at <= issued_at'):
    """Build a deterministic run manifest; split values contain match IDs."""
    matches = [dict(row) for row in matches]
    if not matches:
        raise ValueError('matches must not be empty')
    game_ids = [row.get('game_id') for row in matches]
    if any(game_id is None for game_id in game_ids) or len(set(game_ids)) != len(game_ids):
        raise ValueError('game_id values must be present and unique')
    match_ids = [_match_id(row) for row in matches]
    match_group_ids = sorted(set(match_ids))
    if not isinstance(split, Mapping):
        raise ValueError('split must map train/validation/test to match IDs')
    split_ids = {name: list(split.get(name, ())) for name in ('train', 'validation', 'test')}
    flattened = [str(item) for values in split_ids.values() for item in values]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(match_group_ids):
        raise ValueError('split must assign every match exactly once')
    if not horizons or any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in horizons):
        raise ValueError('horizons must be positive integer seconds')
    ordered_games = sorted(matches, key=lambda row: (row.get('competition_order', 0), str(row['game_id'])))
    ordered_match_ids = list(dict.fromkeys(_match_id(row) for row in ordered_games))
    return {
        'schema_version': 1,
        'all_game_ids': [row['game_id'] for row in ordered_games],
        'all_match_ids': ordered_match_ids,
        'competition_order': {str(row['game_id']): row.get('competition_order') for row in ordered_games},
        'split': {name: [str(match_id) for match_id in values] for name, values in split_ids.items()},
        'split_pass': True,
        'horizons_seconds': sorted(set(horizons)),
        'sides': list(SIDES),
        'visibility_provider': visibility_provider,
        'region_map_version': region_map_version,
        'target_builder_version': target_builder_version,
        'label_version': label_version,
        'feature_cutoff': feature_cutoff,
        'target_cutoff': 'target_at = issued_at + horizon; exact timestamp only',
        'source_mode': 'visibility_masked_replay' if visibility_provider else 'oracle_replay',
        'ordered_game_ids_sha256': _digest([row['game_id'] for row in ordered_games]),
        'manifest_sha256': _digest({
            'games': [(row['game_id'], _match_id(row), row.get('competition_order')) for row in ordered_games],
            'split': split_ids,
            'horizons': sorted(set(horizons)),
            'provider': visibility_provider,
            'labels': label_version,
        }),
    }


def summarize_readiness(manifest, feature_audit, prediction_audit, scores):
    """Summarize evidence and keep deployment blocked unless every gate passes."""
    for name, value in (('manifest', manifest), ('feature_audit', feature_audit),
                        ('prediction_audit', prediction_audit), ('scores', scores)):
        if not isinstance(value, Mapping):
            raise ValueError(f'{name} must be a mapping')
    future_count = feature_audit.get('future_feature_count')
    split_overlap_count = feature_audit.get('split_overlap_count')
    causal_input_pass = (isinstance(future_count, int) and not isinstance(future_count, bool)
                         and future_count == 0)
    split_pass = (manifest.get('split_pass') is True
                  and isinstance(split_overlap_count, int)
                  and not isinstance(split_overlap_count, bool) and split_overlap_count == 0)
    observable_input_pass = (bool(manifest.get('visibility_provider'))
                             and manifest.get('source_mode') == 'visibility_masked_replay')
    invalid_count = prediction_audit.get('invalid_count')
    probability_schema_pass = (isinstance(invalid_count, int) and not isinstance(invalid_count, bool)
                               and invalid_count == 0
                               and prediction_audit.get('probability_schema_pass', True) is True)
    red_blue_complete = prediction_audit.get('red_blue_complete') is True
    side_macro = scores.get('side_macro_average')
    pooled_micro = scores.get('pooled_micro')
    deployment_ready = all((causal_input_pass, observable_input_pass, split_pass,
                            probability_schema_pass, red_blue_complete))
    return {
        'schema_version': 1,
        'causal_input_pass': causal_input_pass,
        'observable_input_pass': observable_input_pass,
        'split_pass': split_pass,
        'probability_schema_pass': probability_schema_pass,
        'red_blue_complete': red_blue_complete,
        'side_macro_scores': side_macro,
        'pooled_micro_scores': pooled_micro,
        'source_mode': manifest.get('source_mode', 'unknown'),
        'deployment_ready': deployment_ready,
        'blocking_reasons': [
            name for name, passed in (
                ('causal_input', causal_input_pass),
                ('observable_input', observable_input_pass),
                ('split_integrity', split_pass),
                ('probability_schema', probability_schema_pass),
                ('red_blue_coverage', red_blue_complete),
            ) if not passed
        ],
    }
