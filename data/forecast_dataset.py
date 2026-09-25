"""Build both-side forecast examples from radar-masked inputs and global labels."""
from collections import defaultdict
from math import isfinite

from data.forecast_targets import build_side_examples
from data.radar_visibility_replay import PROVIDER_ID, build_radar_visibility_snapshots


DATASET_VERSION = 'radar-masked-examples-v1'


def _read(row, field):
    return row[field]


def _target_entity(row):
    def number(field):
        value = _read(row, field)
        return float(value) if value is not None else None

    return {
        'entity_id': f"{_read(row, '阵营')}:{_read(row, 'robot_id')}",
        'team': _read(row, '阵营'),
        'entity_type': _read(row, '机器人类型'),
        'position': [number('x'), number('y'), number('z')],
        'hp': number('当前血量'),
        'maxhp': number('最大血量'),
        'yaw': number('枪口朝向'),
        'power': number('底盘功率'),
        'small_heat': number('小热量'),
        'big_heat': number('大热量'),
        'ammo17': number('累计17mm发弹'),
        'ammo42': number('累计42mm发弹'),
        'radar_highlighted': bool(_read(row, '是否易伤')),
    }


def build_game_forecast_examples(rows, game_id, *, horizons=(10, 30, 60, 120), step=3):
    """Build causal three-second issue points and exact future truth labels.

    Returns red and blue views separately. Hidden enemy measurements are
    omitted from ``observable``; global future rows appear only under targets.
    """
    rows = list(rows)
    if not rows:
        return {'红': [], '蓝': []}
    if isinstance(step, bool) or not isinstance(step, (int, float)) or not isfinite(step) or step <= 0:
        raise ValueError('step must be a positive finite number')
    if any(_read(row, 'game_id') != game_id for row in rows):
        raise ValueError('rows contain a different game_id')
    by_time = defaultdict(list)
    for row in rows:
        by_time[float(_read(row, '时刻秒'))].append(row)

    truth_by_time = {}
    for timestamp, state_rows in by_time.items():
        entities = {}
        teams = {}
        for row in state_rows:
            entity = _target_entity(row)
            entities[entity['entity_id']] = entity
            teams.setdefault(entity['team'], {
                'coins_total': float(_read(row, '队伍总金币') or 0.0),
                'coins_left': float(_read(row, '队伍剩余金币') or 0.0),
            })
        truth_by_time[timestamp] = {
            'target_at': timestamp,
            'entities': entities,
            'teams': teams,
        }

    snapshots = build_radar_visibility_snapshots(rows, game_id=game_id)
    observed_times = sorted(by_time)
    origin = observed_times[0]
    selected = {
        side: [snapshot for snapshot in side_snapshots
               if abs((snapshot['issued_at'] - origin) / step
                      - round((snapshot['issued_at'] - origin) / step)) <= 1e-9]
        for side, side_snapshots in snapshots.items()
    }
    return {
        side: build_side_examples(game_id, selected[side], truth_by_time, side, horizons)
        for side in ('红', '蓝')
    }


def dataset_metadata():
    return {
        'dataset_version': DATASET_VERSION,
        'visibility_provider': PROVIDER_ID,
        'visibility_semantics': 'RMUC 2026 radar protocol fields from replay; opponent robot type/XY/HP/team coins/highlighted state; zero-zero XY means unsent',
        'unsupported_opponent_fields': ['z', 'yaw', 'power', 'heat', 'cumulative_ammo', 'maxhp', 'drone_hp', 'buildings'],
        'unavailable_protocol_fields': ['remaining_projectile_allowance', 'full_buff_packet'],
        'time_resolution_seconds': 1.0,
        'issue_step_seconds': 3.0,
        'feature_cutoff': 'observed_at <= issued_at and received_at <= issued_at; no future fill',
        'target_cutoff': 'target_at = issued_at + horizon; exact timestamp only',
        'sides': ['红', '蓝'],
        'horizons_seconds': [10, 30, 60, 120],
        'global_replay_fields': 'targets only; never copied into observable input',
    }
