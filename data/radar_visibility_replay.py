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
    'z': 'z',
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


def _life_state_from_hp(hp):
    """Infer lifecycle only from an actual numeric HP reading, never a bool."""
    if (isinstance(hp, (int, float)) and not isinstance(hp, bool)
            and isfinite(hp)):
        return 'destroyed' if hp <= 0 else 'active'
    return 'unknown'


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
    previous_ammo = {}
    previous_positions = {}
    # Maintain separate causal display anchors for each observer.  In
    # particular, opponent aircraft HP is withheld, so its hidden death must
    # never change the position or lifecycle exposed to the other side.
    last_visible_positions = {'红': {}, '蓝': {}}
    destroyed_position_anchors = {'红': {}, '蓝': {}}
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
            current_ammo = []
            for field in ('累计17mm发弹', '累计42mm发弹'):
                value = _read(row, field)
                if isinstance(value, (int, float)):
                    current_ammo.append(float(value))
            ammo_delta = None
            if current_ammo:
                key = _entity_id(row)
                previous = previous_ammo.get(key)
                if previous is not None and len(previous) == len(current_ammo):
                    ammo_delta = round(sum(max(0.0, cur - old)
                                           for cur, old in zip(current_ammo, previous)), 4)
                previous_ammo[key] = current_ammo
            if position is not None:
                raw['position'] = position
                # Velocity direction is causal: it is computed only from the
                # last position already received for this robot.  It is
                # available for both teams because radar supplies XY, while
                # the opponent's instantaneous yaw remains withheld.
                motion_key = _entity_id(row)
                previous = previous_positions.get(motion_key)
                if previous is not None and issued_at > previous['t']:
                    dt = float(issued_at - previous['t'])
                    vx = (position[0] - previous['position'][0]) / dt
                    vy = (position[1] - previous['position'][1]) / dt
                    speed = (vx * vx + vy * vy) ** 0.5
                    raw['velocity_x'] = round(vx, 4)
                    raw['velocity_y'] = round(vy, 4)
                    raw['speed_mps'] = round(speed, 4)
                    if speed > 1e-6:
                        import math
                        raw['motion_direction_deg'] = round(
                            math.degrees(math.atan2(vy, vx)), 3)
                previous_positions[motion_key] = {
                    't': float(issued_at), 'position': list(position)}
            for side in ('红', '蓝'):
                if side == team:
                    own = dict(raw)
                    own.update({key: _read(row, col) for key, col in _SELF_FIELDS.items()})
                    # HP is an explicit, current sensor field.  Keep lifecycle
                    # separate from observability: a destroyed robot still has
                    # a measured last position and should be rendered gray.
                    hp = own.get('hp')
                    own['life_state'] = _life_state_from_hp(hp)
                    if ammo_delta is not None:
                        own['ammo_delta'] = ammo_delta
                        own['firing_inferred'] = bool(ammo_delta > 0.0)
                    observations_by_side[side].append(own)
                elif entity_type in _OPPONENT_ROBOT_TYPES:
                    # Radar transmits these fields for opponent robot slots.
                    # The replay lacks remaining ammo and current buff packets.
                    enemy = {key: value for key, value in raw.items()
                             if key in {'entity_id', 'team', 'entity_type',
                                        'observed_at', 'received_at', 'confidence',
                                        'position', 'velocity_x', 'velocity_y',
                                        'speed_mps', 'motion_direction_deg'}}
                    enemy['source'] = 'radar'
                    enemy.update({key: _read(row, col) for key, col in _OPPONENT_FIELDS.items()
                                  if key != 'hp' or entity_type != '空中'})
                    # Opponent air HP is not present in the radar contract, so
                    # its death state must remain unknown from this viewpoint.
                    hp = enemy.get('hp')
                    enemy['life_state'] = _life_state_from_hp(hp)
                    enemy['radar_highlighted'] = bool(_read(row, '是否易伤'))
                    if position is not None:
                        enemy['position'] = position
                    observations_by_side[side].append(enemy)

        for side in ('红', '蓝'):
            snapshot = build_observable_snapshot(
                observations_by_side[side], issued_at, side,
                {'sensor': TIME_RESOLUTION_SECONDS, 'radar': TIME_RESOLUTION_SECONDS},
            )
            for entity_id, entity in snapshot['entities'].items():
                if entity.get('observability_state') != 'current':
                    continue
                life_state = entity.get('life_state', 'unknown')
                position = entity.get('position')
                anchors = destroyed_position_anchors[side]
                last_positions = last_visible_positions[side]
                if life_state == 'destroyed':
                    if entity_id not in anchors:
                        anchor = position
                        source = 'first_confirmed_destroyed_observation'
                        if anchor is None:
                            previous = last_positions.get(entity_id)
                            if previous and issued_at - previous[0] <= TIME_RESOLUTION_SECONDS + 1e-6:
                                anchor = previous[1]
                                source = 'recent_position_before_destroyed'
                        anchors[entity_id] = (list(anchor) if anchor is not None else None, source)
                    anchor, source = anchors[entity_id]
                    entity['life_state_source'] = 'current_hp_zero'
                    if anchor is not None:
                        entity['position'] = list(anchor)
                        entity['position_anchor_source'] = source
                    else:
                        entity.pop('position', None)
                    # A dead unit has no meaningful movement direction, even
                    # if noisy source XY keeps changing after HP reaches zero.
                    for key in ('velocity_x', 'velocity_y', 'speed_mps',
                                'motion_direction_deg'):
                        entity.pop(key, None)
                elif life_state == 'unknown' and entity_id in anchors:
                    # A missing HP sample cannot undo an already confirmed
                    # death.  Only an explicit positive HP reading (respawn)
                    # starts a new live track.
                    anchor, source = anchors[entity_id]
                    entity['life_state'] = 'destroyed'
                    entity['life_state_source'] = 'prior_confirmed_hp_zero'
                    if anchor is not None:
                        entity['position'] = list(anchor)
                        entity['position_anchor_source'] = source
                    else:
                        entity.pop('position', None)
                    for key in ('velocity_x', 'velocity_y', 'speed_mps',
                                'motion_direction_deg'):
                        entity.pop(key, None)
                else:
                    was_destroyed = entity_id in anchors
                    if life_state == 'active':
                        anchors.pop(entity_id, None)
                    if was_destroyed:
                        last_positions.pop(entity_id, None)
                        # The first live position after respawn is a new track;
                        # a velocity over the death interval is meaningless.
                        for key in ('velocity_x', 'velocity_y', 'speed_mps',
                                    'motion_direction_deg'):
                            entity.pop(key, None)
                    if position is not None:
                        last_positions[entity_id] = (issued_at, list(position))
            snapshot['visibility_provider'] = PROVIDER_ID
            snapshot['visibility_semantics'] = 'protocol-fields-only; zero-zero opponent position means unsent'
            snapshot['protocol_version'] = PROTOCOL_VERSION
            snapshot['allowed_opponent_fields'] = ['entity_type', 'position_xy', 'hp', 'team_coins_total', 'team_coins_left', 'radar_highlighted']
            snapshot['allowed_self_sensor_fields'] = [
                'position_xyz', 'hp', 'maxhp', 'yaw', 'power', 'heat',
                'cumulative_ammo', 'team_coins_total', 'team_coins_left',
                'ammo_delta', 'firing_inferred', 'velocity_x', 'velocity_y',
                'speed_mps', 'motion_direction_deg',
            ]
            snapshot['derived_motion_fields'] = [
                'velocity_x', 'velocity_y', 'speed_mps', 'motion_direction_deg'
            ]
            snapshot['withheld_opponent_fields'] = ['z', 'yaw', 'power', 'heat', 'cumulative_ammo', 'maxhp', 'remaining_ammo', 'buffs', 'building_details']
            snapshot['time_resolution_seconds'] = TIME_RESOLUTION_SECONDS
            result[side].append(snapshot)
    return result


def load_radar_visibility_snapshots(con, game_id):
    """Load one match's official replay rows and apply the visibility mask."""
    return build_radar_visibility_snapshots(timeseries_for(con, game_id), game_id=game_id)
