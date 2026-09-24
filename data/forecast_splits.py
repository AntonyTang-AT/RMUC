"""Chronological, whole-match train/validation/test partitions."""

from collections.abc import Mapping
from datetime import datetime, timedelta
from math import isfinite


def _row_mapping(row):
    # sqlite3.Row supplies keys() and item access, but is not a Mapping.
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, 'keys'):
        return {key: row[key] for key in row.keys()}
    raise ValueError('match row must be a mapping')


def _match_id(row):
    explicit = row.get('match_id')
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    region = row.get('赛区', row.get('region'))
    number = row.get('场次号', row.get('match_number'))
    if region is None or number is None or not str(region).strip():
        raise ValueError('match_id missing; need 赛区 and 场次号 to reconstruct it')
    return f'{region}:{number}'


def _start_time(row):
    raw = row.get('开始时间', row.get('start_time'))
    if isinstance(raw, datetime):
        return raw.isoformat()
    if isinstance(raw, str) and raw.strip():
        try:
            return datetime.fromisoformat(raw.strip()).isoformat()
        except ValueError as exc:
            raise ValueError(f'invalid 开始时间: {raw!r}') from exc
    raise ValueError('match row missing 开始时间/start_time')


def _end_time(row, start):
    raw_end = row.get('结束时间', row.get('end_time'))
    if raw_end is not None and str(raw_end).strip():
        if isinstance(raw_end, datetime):
            end = raw_end
        elif isinstance(raw_end, str):
            try:
                end = datetime.fromisoformat(raw_end.strip())
            except ValueError as exc:
                raise ValueError(f'invalid 结束时间: {raw_end!r}') from exc
        else:
            raise ValueError('结束时间/end_time must be a datetime or ISO string')
    else:
        duration = row.get('时长秒', row.get('duration_seconds', row.get('duration')))
        if duration is None:
            raise ValueError('match row needs 结束时间/end_time or 时长秒/duration_seconds')
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not isfinite(duration) or duration < 0):
            raise ValueError('时长秒/duration_seconds must be a nonnegative finite number')
        end = datetime.fromisoformat(start) + timedelta(seconds=duration)
    start_dt = datetime.fromisoformat(start)
    if end < start_dt:
        raise ValueError('match end time cannot precede start time')
    return end.isoformat()


def chronological_group_split(match_rows, train_fraction, validation_fraction,
                              *, sensor_provider_version=None,
                              horizons=(10, 30, 60, 120)):
    """Assign every game and both future side views to one ordered match split."""
    for name, fraction in [('train_fraction', train_fraction),
                           ('validation_fraction', validation_fraction)]:
        if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not isfinite(fraction) or fraction < 0 or fraction >= 1):
            raise ValueError(f'{name} must be in [0, 1)')
    if train_fraction + validation_fraction >= 1:
        raise ValueError('train and validation fractions must leave a test partition')

    groups = {}
    seen_game_ids = set()
    for source_row in match_rows:
        row = _row_mapping(source_row)
        game_id = row.get('game_id')
        if game_id is None or game_id in seen_game_ids:
            raise ValueError('game_id must be present and unique across match rows')
        seen_game_ids.add(game_id)
        match_id = _match_id(row)
        start = _start_time(row)
        end = _end_time(row, start)
        group = groups.setdefault(match_id, {'match_id': match_id, 'rows': [], 'starts': [], 'ends': [],
                                             'team_pairs': set()})
        group['rows'].append(row)
        group['starts'].append(start)
        group['ends'].append(end)
        red_school = row.get('红方学校', row.get('red_school'))
        blue_school = row.get('蓝方学校', row.get('blue_school'))
        if red_school and blue_school:
            group['team_pairs'].add(frozenset((red_school, blue_school)))
            if len(group['team_pairs']) > 1:
                raise ValueError(f'match_id {match_id!r} identifies multiple team pairs')
    ordered = sorted(groups.values(), key=lambda group: (min(group['starts']), group['match_id']))
    count = len(ordered)
    train_end = int(count * train_fraction)
    validation_end = train_end + int(count * validation_fraction)
    partitions = {'train': ordered[:train_end],
                  'validation': ordered[train_end:validation_end],
                  'test': ordered[validation_end:]}
    for earlier, later in (('train', 'validation'), ('validation', 'test'),
                           ('train', 'test')):
        if partitions[earlier] and partitions[later]:
            latest_earlier = max(max(group['ends']) for group in partitions[earlier])
            earliest_later = min(min(group['starts']) for group in partitions[later])
            if latest_earlier >= earliest_later:
                raise ValueError(f'{earlier}/{later} match times overlap across split boundary')

    result = {}
    manifest = {
        'split_unit': 'complete_match',
        'order': 'earliest_start_time_then_match_id',
        'horizons_seconds': list(horizons),
        'feature_cutoff': 'observed_at <= issued_at and received_at <= issued_at',
        'target_cutoff': 'target_at = issued_at + horizon, exact timestamp only',
        'sensor_provider_version': sensor_provider_version,
        'all_match_ids': [group['match_id'] for group in ordered],
        'all_game_ids': [row['game_id'] for group in ordered
                         for row in sorted(group['rows'], key=lambda row: _start_time(row))],
    }
    for split_name, partition in partitions.items():
        result[split_name] = [group['match_id'] for group in partition]
        result[f'{split_name}_order'] = [ordered.index(group) for group in partition]
        game_rows = [row for group in partition for row in group['rows']]
        game_rows.sort(key=lambda row: (_start_time(row), row['game_id']))
        manifest[split_name] = {
            'match_ids': result[split_name],
            'game_ids': [row['game_id'] for row in game_rows],
            'start_time': min((_start_time(row) for row in game_rows), default=None),
            'end_time': max((_end_time(row, _start_time(row)) for row in game_rows), default=None),
        }
    result['manifest'] = manifest
    return result
