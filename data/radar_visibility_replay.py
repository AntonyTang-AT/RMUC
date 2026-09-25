"""Replay the fields available through the RMUC 2026 referee/radar protocol.

The official 2026 communication protocol sends opponent robot XY, HP, ammo,
team macro-state, and buffs to radar. The project replay has XY, HP, and team
coins, but lacks opponent ammo allowance and the full buff packet. We expose
only those available protocol fields. A (0, 0) position is explicitly treated
as unsent; opponent height, heading, power, heat, and cumulative shots are
withheld because the radar protocol does not provide them.
"""
from collections import defaultdict
from collections.abc import Mapping
from math import isfinite

from data.observable import build_observable_snapshot
from data.db import timeseries_for


PROVIDER_ID = 'rmuc_2026_referee_radar_protocol_v1'
PROTOCOL_VERSION = 'RMUC-2026-Communication-Protocol-v1.3.0'
TIME_RESOLUTION_SECONDS = 1.0

_SELF_FIELDS = {
    'hp': '当前血量',
    'maxhp': '最大血量',
    'yaw': '枪口朝向',
    'power': '底盘功率',
    'small_heat': '小热量',
    'big_heat': '大热量',
    'ammo17': '累计17mm发弹',
    'ammo42': '累计42mm发弹',
    'team_coins_total': '队伍总金币',
    'team_coins_left': '队伍剩余金币',
}

_OPPONENT_FIELDS = {
    'hp': '当前血量',
    'team_coins_total': '队伍总金币',
    'team_coins_left': '队伍剩余金币',
}
_OPPONENT_ROBOT_TYPES = frozenset({'英雄', '工程', '步兵3', '步兵4', '空中', '哨兵'})


def _read(row, name):
    if isinstance(row, Mapping):
        return row[name]
    return row[name]


def _entity_id(row):
    return f"{_read(row, '阵营')}:{_read(row, 'robot_id')}"


def build_radar_visibility_snapshots(rows, game_id=None):
    """Return ``{'红': [...], '蓝': [...]}`` snapshots from causal replay rows.

    At each observed second, own-team state is retained as self telemetry.
    Opponent robot rows expose XY, current HP, team coins, robot class, and
    radar-highlight status. Opponent buildings and unsupported fine-grained
    fields are omitted. No forward filling or nearest-future lookup is done.
    """
    rows = list(rows)
    if not rows:
        return {'红': [], '蓝': []}
    by_time = defaultdict(list)
    for row in rows:
        timestamp = _read(row, '时刻秒')
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not isfinite(timestamp) or timestamp < 0):
            raise ValueError('时刻秒 must be a nonnegative number')
        if game_id is not None and _read(row, 'game_id') != game_id:
            raise ValueError('rows contain a different game_id')
        if _read(row, '阵营') not in ('红', '蓝'):
            raise ValueError('阵营 must be 红 or 蓝')
        by_time[float(timestamp)].append(row)

    result = {'红': [], '蓝': []}
    for issued_at in sorted(by_time):
        at_time = by_time[issued_at]
        observations_by_side = {'红': [], '蓝': []}
        for row in at_time:
            team = _read(row, '阵营')
            entity_type = _read(row, '机器人类型')
            x, y = _read(row, 'x'), _read(row, 'y')
            valid_xy = (
                x is not None and y is not None
                and not (isinstance(x, bool) or isinstance(y, bool))
                and isfinite(float(x)) and isfinite(float(y))
                and not (float(x) == 0 and float(y) == 0)
            )
            position = [float(x), float(y)] if valid_xy else None
            raw = {
                'entity_id': _entity_id(row),
                'team': team,
                'entity_type': entity_type,
                'observed_at': issued_at,
                'received_at': issued_at,
                'source': 'sensor',
                'confidence': 1.0,
            }
            if position is not None:
                raw['position'] = position
            for side in ('红', '蓝'):
                if side == team:
                    own = dict(raw)
                    own.update({key: _read(row, col) for key, col in _SELF_FIELDS.items()})
                    observations_by_side[side].append(own)
                elif entity_type in _OPPONENT_ROBOT_TYPES:
                    # Radar transmits these fields for opponent robot slots.
                    # The replay lacks remaining ammo and current buff packets.
                    enemy = {key: value for key, value in raw.items()
                             if key in {'entity_id', 'team', 'entity_type',
                                        'observed_at', 'received_at', 'confidence'}}
                    enemy['source'] = 'radar'
                    enemy.update({key: _read(row, col) for key, col in _OPPONENT_FIELDS.items()
                                  if key != 'hp' or entity_type != '空中'})
                    enemy['radar_highlighted'] = bool(_read(row, '是否易伤'))
                    if position is not None:
                        enemy['position'] = position
                    observations_by_side[side].append(enemy)

        for side in ('红', '蓝'):
            snapshot = build_observable_snapshot(
                observations_by_side[side], issued_at, side,
                {'sensor': TIME_RESOLUTION_SECONDS, 'radar': TIME_RESOLUTION_SECONDS},
            )
            snapshot['visibility_provider'] = PROVIDER_ID
            snapshot['visibility_semantics'] = 'protocol-fields-only; zero-zero opponent position means unsent'
            snapshot['protocol_version'] = PROTOCOL_VERSION
            snapshot['allowed_opponent_fields'] = ['entity_type', 'position_xy', 'hp', 'team_coins_total', 'team_coins_left', 'radar_highlighted']
            snapshot['withheld_opponent_fields'] = ['z', 'yaw', 'power', 'heat', 'cumulative_ammo', 'maxhp', 'remaining_ammo', 'buffs', 'building_details']
            snapshot['time_resolution_seconds'] = TIME_RESOLUTION_SECONDS
            result[side].append(snapshot)
    return result


def load_radar_visibility_snapshots(con, game_id):
    """Load one match's official replay rows and apply the visibility mask."""
    return build_radar_visibility_snapshots(timeseries_for(con, game_id), game_id=game_id)
