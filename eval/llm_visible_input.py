"""Construct one LLM decision input from causal sensor/radar replay snapshots.

The caller supplies snapshots made by ``data.radar_visibility_replay``.  Only
the selected side's current and preceding frames are read.  A positive-list
projection is applied again here so additions to the replay schema cannot
silently enter a model prompt.  No future outcome is returned by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

from data.radar_visibility_replay import PROVIDER_ID
from schema import layered as sl
from worldstate.functional_zone import assign_functional_zone

INPUT_CONTRACT = 'rmuc_2026_referee_radar_protocol_v1'
_TEAMS = {'红', '蓝'}
_ROBOT_TYPES = {'英雄', '工程', '步兵3', '步兵4', '空中', '哨兵'}
_OWN_BUILDINGS = {'基地', '前哨站'}
_COMMON = ('entity_id', 'team', 'entity_type', 'role', 'source',
           'observability_state', 'presence_state', 'observed_at',
           'received_at', 'age_seconds', 'confidence', 'life_state',
           'position', 'velocity_x', 'velocity_y', 'speed_mps',
           'motion_direction_deg')
_OWN = ('z', 'hp', 'maxhp', 'yaw', 'power', 'small_heat', 'big_heat',
        'ammo17', 'ammo42', 'team_coins_total', 'team_coins_left',
        'ammo_delta', 'firing_inferred')
_ENEMY = ('hp', 'team_coins_total', 'team_coins_left', 'radar_highlighted')
_MEASURED = set(_OWN + _ENEMY + ('position', 'velocity_x', 'velocity_y',
                                 'speed_mps', 'motion_direction_deg'))


def conservative_slot_action(status='unseen'):
    """Return the visibility-safe action for a non-actionable own slot."""
    return {
        'target': '无',
        'move': '无',
        'fire': False,
        'risk': '低' if status == 'destroyed' else '高',
    }


def _explicitly_destroyed(entity):
    if not isinstance(entity, dict):
        return False
    if entity.get('life_state') == 'destroyed':
        return True
    hp_ratio = entity.get('hp_ratio')
    return (isinstance(hp_ratio, (int, float)) and not isinstance(hp_ratio, bool)
            and float(hp_ratio) <= 0.0)


def own_unit_slot_states(record):
    """Describe all fixed output slots using only issue-time visible state."""
    side = record.get('perspective')
    if side not in _TEAMS:
        raise ValueError('visible record has no valid perspective')
    robots = (record.get('issue_state') or {}).get('robots') or {}
    output = {}
    for unit_type in sl.UNIT_CLASSES:
        candidates = [
            (str(entity_id), entity) for entity_id, entity in robots.items()
            if entity.get('team') == side and entity.get('type') == unit_type
        ]
        current = [item for item in candidates
                   if item[1].get('observability_state', 'current') == 'current'
                   and not _explicitly_destroyed(item[1])]
        destroyed = [item for item in candidates
                     if _explicitly_destroyed(item[1])]
        if len(current) == 1:
            entity_id, entity = current[0]
            output[unit_type] = {'status': 'current', 'entity_id': entity_id,
                                 'entity': entity}
        elif len(current) > 1:
            output[unit_type] = {'status': 'ambiguous', 'entity_id': None,
                                 'entity': None}
        elif destroyed:
            output[unit_type] = {'status': 'destroyed', 'entity_id': None,
                                 'entity': None}
        else:
            output[unit_type] = {'status': 'unseen', 'entity_id': None,
                                 'entity': None}
    return output


def observable_target_types(record):
    """Return the opponent robot classes confirmed by the issue-time radar view."""
    side = record.get('perspective')
    return {
        entity.get('type') for entity in ((record.get('issue_state') or {}).get('robots') or {}).values()
        if entity.get('team') in _TEAMS and entity.get('team') != side
        and entity.get('observability_state', 'current') == 'current'
        and entity.get('life_state') != 'destroyed'
    }


def project_decision_to_observable_contract(decision, record):
    """Remove label dependence on hidden own units and unobserved enemy classes."""
    projected = deepcopy(decision)
    if not isinstance(projected, dict) or not isinstance(projected.get('units'), dict):
        return projected
    slots = own_unit_slot_states(record)
    observed_targets = observable_target_types(record)
    target_to_type = {
        '敌方英雄': '英雄',
        '敌方工程': '工程',
        '敌方哨兵': '哨兵',
    }
    for unit_type in sl.UNIT_CLASSES:
        action = projected['units'].get(unit_type)
        status = slots[unit_type]['status']
        if status != 'current':
            projected['units'][unit_type] = conservative_slot_action(status)
            continue
        if not isinstance(action, dict):
            continue
        target = action.get('target')
        target_type = target_to_type.get(target)
        if target == '敌方步兵':
            target_type = '步兵3' if '步兵3' in observed_targets else (
                '步兵4' if '步兵4' in observed_targets else None)
        if target in target_to_type or target == '敌方步兵':
            if target_type is None or target_type not in observed_targets:
                action['target'] = '无'
                action['fire'] = False
    return projected


def mask_own_unit_slot(record, unit_type):
    """Create a causal counterfactual where one own unit is intentionally unseen."""
    if unit_type not in sl.UNIT_CLASSES:
        raise ValueError(f'unknown unit slot: {unit_type}')
    masked = deepcopy(record)
    side = masked.get('perspective')
    if side not in _TEAMS:
        raise ValueError('visible record has no valid perspective')
    for frame in masked.get('visible_history') or ():
        entities = frame.get('entities') or {}
        for entity_id in [
                key for key, entity in entities.items()
                if entity.get('team') == side and entity.get('entity_type') == unit_type]:
            entities.pop(entity_id, None)
    issue_robots = (masked.get('issue_state') or {}).get('robots') or {}
    for entity_id in [
            key for key, entity in issue_robots.items()
            if entity.get('team') == side and entity.get('type') == unit_type]:
        issue_robots.pop(entity_id, None)
    prompt = _render_prompt(masked.get('visible_history') or (), side)
    masked['prompt_text'] = prompt
    masked['input_sha256'] = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    audit = dict(masked.get('input_audit') or {})
    audit['counterfactual_visibility_mask'] = unit_type
    masked['input_audit'] = audit
    return masked


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _project_entity(entity, side, frame_time):
    if not isinstance(entity, dict):
        raise ValueError('entity must be a mapping')
    team = entity.get('team')
    typ = entity.get('entity_type')
    if team not in _TEAMS or (typ not in _ROBOT_TYPES and
                             not (team == side and typ in _OWN_BUILDINGS)):
        return None
    own = team == side
    if entity.get('role') != ('self' if own else 'opponent'):
        raise ValueError('role/team mismatch in visibility snapshot')
    if entity.get('source') != ('sensor' if own else 'radar'):
        raise ValueError('sensor/radar source mismatch in visibility snapshot')
    for key in ('observed_at', 'received_at'):
        if _number(entity.get(key), key) > frame_time:
            raise ValueError('future measurement in visibility snapshot')
    if entity['received_at'] < entity['observed_at']:
        raise ValueError('received_at precedes observed_at')
    allowed = (*_COMMON, *(_OWN if own else _ENEMY))
    projected = {key: entity[key] for key in allowed if key in entity}
    if projected.get('observability_state') != 'current':
        for key in _MEASURED:
            projected.pop(key, None)
    pos = projected.get('position')
    if pos is not None:
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            raise ValueError('position must be XY')
        projected['position'] = [_number(pos[0], 'x'), _number(pos[1], 'y')]
    if not own and team != side:
        # Opponent aircraft HP is withheld even if an upstream provider changes.
        if entity['entity_type'] == '空中':
            projected.pop('hp', None)
    return projected


def _project_snapshot(snapshot, side):
    if snapshot.get('visibility_provider') != PROVIDER_ID or snapshot.get('self_team') != side:
        raise ValueError('official side-specific visibility provider required')
    frame_time = _number(snapshot.get('issued_at'), 'issued_at')
    entities = snapshot.get('entities')
    if not isinstance(entities, dict):
        raise ValueError('entities must be a mapping')
    safe = {}
    for key, entity in entities.items():
        value = _project_entity(entity, side, frame_time)
        if value is not None:
            if str(value.get('entity_id')) != str(key):
                raise ValueError('entity key/id mismatch')
            safe[str(key)] = value
    return {'t': frame_time, 'side': side, 'entities': safe}


def _make_issue_state(current):
    robots = {}
    buildings = {}
    for key, entity in current['entities'].items():
        item = dict(entity)
        item['type'] = item.pop('entity_type')
        pos = item.get('position')
        if pos is not None:
            item['functional_zone'] = assign_functional_zone(*pos)
        hp, maxhp = item.get('hp'), item.get('maxhp')
        if isinstance(hp, (int, float)) and isinstance(maxhp, (int, float)) and maxhp > 0:
            item['hp_ratio'] = max(0.0, min(1.0, hp / maxhp))
        if item['type'] in _OWN_BUILDINGS:
            buildings[key] = item
        else:
            robots[key] = item
    return {'t': current['t'], 'robots': robots, 'buildings': buildings}


def _prompt_context(history, side):
    """Compact, deterministic rendering of already projected visible fields."""
    frames = []
    for index, frame in enumerate(history):
        latest = index == len(history) - 1
        entities = []
        for entity_id, entity in sorted(frame['entities'].items()):
            row = {'id': entity_id, '兵种': entity['entity_type']}
            if entity.get('observability_state') != 'current':
                row['可见性'] = entity.get('observability_state', 'unknown')
                entities.append(row)
                continue
            pos = entity.get('position')
            if pos is not None:
                row['位置'] = [round(v, 2) for v in pos]
            if latest:
                for key, label in (('life_state', '存活'), ('hp', '血量'),
                                   ('speed_mps', '速度'), ('motion_direction_deg', '运动方向')):
                    if entity.get(key) is not None:
                        row[label] = entity[key]
                if entity['team'] == side:
                    for key, label in (('yaw', '朝向'), ('firing_inferred', '发弹'),
                                       ('small_heat', '小热量'), ('big_heat', '大热量')):
                        if entity.get(key) is not None:
                            row[label] = entity[key]
                elif entity.get('radar_highlighted') is not None:
                    row['雷达标记'] = entity['radar_highlighted']
            entities.append(row)
        compact_frame = {'时刻': frame['t'], '可见目标': entities}
        if latest:
            own_coins = next((entity.get('team_coins_left')
                              for entity in frame['entities'].values()
                              if entity.get('team') == side
                              and entity.get('team_coins_left') is not None), None)
            if own_coins is not None:
                compact_frame['己方剩余金币'] = own_coins
        frames.append(compact_frame)
    return json.dumps({'我方': side, '历史至当前': frames},
                      ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def _render_prompt(history, side):
    return (f'你是{side}方的比赛战术助手。以下仅是截至当前时刻的己方传感器和敌方雷达记录。'
            '对敌方缺失字段保持未知。请给出当前态势、下一步具体建议及理由；'
            '机器人建议区分单位类型，不能把预测当作已经发生的事实。\n'
            f'可见记录：{_prompt_context(history, side)}')


def format_visible_training_instruction(record, forecast_context=None, *, max_chars=760):
    """Render the compact, auditable instruction used for LoRA training/inference.

    ``build_visible_llm_input`` deliberately preserves a rich four-frame
    audit trail.  That representation is too long to feed directly to the
    896-token Tier2 generation budget, so this function creates a bounded
    *model* view from the already validated issue state.  It never falls back
    to a full replay decision point.  Forecasts remain reference-only and are
    compacted with the same formatter used by the existing temporal adapter.
    """
    if not validate_visible_llm_input(record):
        raise ValueError('visible LLM input failed contract validation')
    try:
        budget = int(max_chars)
    except (TypeError, ValueError) as exc:
        raise ValueError('max_chars must be an integer') from exc
    if budget < 160:
        raise ValueError('max_chars must be at least 160')

    side = record['perspective']
    issue = record['issue_state']
    slots = own_unit_slot_states(record)
    own, opponent = [], []
    # The issue state is a positive-list projection made by this module.
    # Keep the training rendering intentionally terse: the important current
    # facts are unit class, location, health when measured, motion, and own
    # firing/heat signals.  Never synthesize a missing opponent field.
    for entity_id, entity in sorted((issue.get('robots') or {}).items()):
        # The structured output addresses a unit class, not an entity ID.
        # Keep the operationally meaningful visible facts compact enough that
        # every own and opponent unit survives the shared prompt budget.
        fields = [str(entity.get('type', '未知'))]
        zone = entity.get('functional_zone')
        if zone:
            fields.append(f'区={zone}')
        if entity.get('hp_ratio') is not None:
            fields.append(f'血={float(entity["hp_ratio"]):.2f}')
        elif entity.get('hp') is not None:
            fields.append(f'血={float(entity["hp"]):.0f}')
        if entity.get('life_state') and entity.get('life_state') != 'active':
            fields.append(f'态={entity["life_state"]}')
        if entity.get('speed_mps') is not None:
            fields.append(f'速={float(entity["speed_mps"]):.1f}')
        if entity.get('team') == side:
            if entity.get('firing_inferred') is True:
                fields.append('发射')
            if entity.get('small_heat') is not None:
                fields.append(f'热={float(entity["small_heat"]):.0f}')
            own.append('/'.join(fields))
        else:
            opponent.append('/'.join(fields))

    buildings = []
    for entity_id, entity in sorted((issue.get('buildings') or {}).items()):
        fields = [str(entity.get('type', '建筑')), str(entity_id)]
        if entity.get('hp_ratio') is not None:
            fields.append(f'血={float(entity["hp_ratio"]):.2f}')
        elif entity.get('hp') is not None:
            fields.append(f'血={float(entity["hp"]):.0f}')
        buildings.append('/'.join(fields))

    slot_text = ';'.join(
        f'{unit_type}={slots[unit_type]["status"]}' for unit_type in sl.UNIT_CLASSES)
    visible = (
        f"局势: t={float(record['issued_at_s']):.0f} 视角={side} | "
        f"己方固定槽位: {slot_text} | "
        f"己方可见: {';'.join(own) if own else '无'} | "
        f"敌方雷达: {';'.join(opponent) if opponent else '无'}"
    )
    if buildings:
        visible += ' | 己方建筑: ' + ';'.join(buildings)

    # This marker is also the boundary understood by the shared prompt
    # compactor.  Without it, a long visible-state section could be mistaken
    # for the whole instruction and truncate the latter units before the
    # forecast reference is appended.
    visible += ' | 时间尺度: 当前可见事实；预测仅作参考'

    if forecast_context:
        from worldstate.forecast_context import format_forecast_context
        forecast = format_forecast_context(forecast_context, max_chars=6000)
        if forecast:
            visible += ' | ' + forecast.strip()

    # Import lazily to avoid a module-level dependency from the visibility
    # contract into the training stack.  The compactor keeps visible evidence
    # first and forecast head/model-vs-baseline evidence second.
    from finetune.lora_train import compact_training_instruction
    return compact_training_instruction(visible, max_chars=budget)


def build_visible_llm_input(snapshots, *, game_id, side, issued_at_s, history_frames=4):
    """Build an auditable prompt and scorecard-compatible issue state.

    ``snapshots`` may contain later frames; those frames are never inspected.
    An exact frame at ``issued_at_s`` is required.  Each earlier frame must
    originate from the same side-specific visibility provider.
    """
    if side not in _TEAMS:
        raise ValueError('decision side must be 红 or 蓝')
    issued_at_s = _number(issued_at_s, 'issued_at_s')
    if not isinstance(history_frames, int) or not 1 <= history_frames <= 32:
        raise ValueError('history_frames must be an integer in [1, 32]')
    earlier = [snap for snap in snapshots
               if _number(snap.get('issued_at'), 'snapshot issued_at') <= issued_at_s]
    selected = sorted(earlier, key=lambda item: item['issued_at'])[-history_frames:]
    if not selected or float(selected[-1]['issued_at']) != issued_at_s:
        raise ValueError('exact issue-time visibility snapshot is required')
    if len({float(s['issued_at']) for s in selected}) != len(selected):
        raise ValueError('duplicate visibility snapshot timestamp')
    history = [_project_snapshot(s, side) for s in selected]
    current = history[-1]
    issue_state = _make_issue_state(current)
    # The serialized context is exactly the data sent to both model adapters.
    prompt = _render_prompt(history, side)
    return {
        'schema_version': 2, 'game_id': int(game_id), 'issued_at_s': issued_at_s,
        'perspective': side, 'visibility_policy': INPUT_CONTRACT,
        'input_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
        'prompt_text': prompt, 'visible_history': history,
        'issue_state': issue_state,
        'input_audit': {'provider': PROVIDER_ID, 'history_frames': len(history),
                        'latest_frame_exact': True, 'future_rows_used': False},
    }


def validate_visible_llm_input(record):
    """Fail closed before claiming a row was generated from a visible view."""
    if record.get('visibility_policy') != INPUT_CONTRACT:
        return False
    try:
        side = record['perspective']
        frames = record['visible_history']
        if not frames or side not in _TEAMS:
            return False
        t = _number(record['issued_at_s'], 'issued_at_s')
        if frames[-1]['t'] != t or any(frames[i]['t'] >= frames[i + 1]['t']
                                        for i in range(len(frames) - 1)):
            return False
        if any(frame['side'] != side or frame['t'] > t for frame in frames):
            return False
        for frame in frames:
            if set(frame) != {'t', 'side', 'entities'}:
                return False
            for key, entity in frame['entities'].items():
                own = entity['team'] == side
                if entity['entity_id'] != key or entity['role'] != ('self' if own else 'opponent'):
                    return False
                if entity['source'] != ('sensor' if own else 'radar'):
                    return False
                if any(k not in (*_COMMON, *(_OWN if own else _ENEMY)) for k in entity):
                    return False
                if (not own and entity.get('entity_type') == '空中' and 'hp' in entity):
                    return False
                if entity.get('observability_state') != 'current' and any(
                        k in entity for k in _MEASURED):
                    return False
                if entity['observed_at'] > frame['t'] or entity['received_at'] > frame['t']:
                    return False
        if record['issue_state'] != _make_issue_state(frames[-1]):
            return False
        audit = record.get('input_audit') or {}
        if (audit.get('provider') != PROVIDER_ID or audit.get('future_rows_used') is not False
                or audit.get('history_frames') != len(frames)):
            return False
        prompt = record['prompt_text']
        if hashlib.sha256(prompt.encode('utf-8')).hexdigest() != record['input_sha256']:
            return False
        return prompt == _render_prompt(frames, side)
    except (KeyError, TypeError, ValueError):
        return False


def load_visible_llm_input(con, *, game_id, side, issued_at_s, history_frames=4):
    """Read only a replay prefix before making the side-specific LLM input.

    Future telemetry is deliberately absent from this SQL query.  The
    observation adapter computes motion/ammo differences causally over the
    prefix, then the positive-list projection above prepares the prompt.
    """
    from data.radar_visibility_replay import build_radar_visibility_snapshots

    issued_at_s = _number(issued_at_s, 'issued_at_s')
    rows = [dict(row) for row in con.execute(
        'SELECT * FROM timeseries WHERE game_id=? AND "时刻秒"<=? ORDER BY "时刻秒"',
        (int(game_id), issued_at_s))]
    snapshots = build_radar_visibility_snapshots(rows, game_id=int(game_id))[side]
    return build_visible_llm_input(
        snapshots, game_id=game_id, side=side, issued_at_s=issued_at_s,
        history_frames=history_frames)
